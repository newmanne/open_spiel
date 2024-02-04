import logging
from statsmodels.distributions.empirical_distribution import ECDF
from open_spiel.python.examples.ubc_utils import load_game_config
from auctions.models import *
from open_spiel.python.examples.ppo_utils import make_env_and_policy, EnvParams, make_ppo_agent, make_dqn_agent
from open_spiel.python.examples.ubc_plotting_utils import parse_run
import pytz
import datetime
import torch
import pyspiel
from distutils import util
import json
from open_spiel.python.examples.ppo_eval import EvalDefaults
import os
from compress_pickle import dumps, loads

OUTPUT_ROOT = os.environ['CLOCK_AUCTION_OUTPUT_ROOT'] 

def get_or_create_game(game_name):
    try:
        game = Game.objects.get(name=game_name)
        game_config = game.config

        # Do this as a sanity check
        game_config_on_disk = load_game_config(game_name)
        if game_config != game_config_on_disk:
            raise ValueError(f"Game config for {game_name} has changed on disk!")

    except Game.DoesNotExist:
        game_obj = pyspiel.load_game('python_clock_auction', dict(filename=game_name))
        game_config = load_game_config(game_name)
        game, _ = Game.objects.get_or_create( 
            name=game_name,
            num_players=game_obj.num_players(),
            num_actions=game_obj.num_distinct_actions(),
            num_products=len(game_config['licenses']),
            config=game_config
        )

    return game

def env_and_policy_from_run(run, env_params=None):
    game = run.game.load_as_spiel()
    env_and_policy = make_env_and_policy(game, dict(run.config), env_params=env_params)
    return env_and_policy

def env_and_policy_for_dry_run(game_db_obj, config, env_params=None):
    game = game_db_obj.load_as_spiel()
    env_and_policy = make_env_and_policy(game, dict(config), env_params=env_params)
    return env_and_policy

def ppo_db_checkpoint_loader(checkpoint, env_params=None):
    # Create an env_and_policy based on a checkpoint in the database
    env_and_policy = env_and_policy_from_run(checkpoint.equilibrium_solver_run, env_params=env_params)
    solver_type = checkpoint.equilibrium_solver_run.config.get('solver_type', 'ppo')
    if solver_type == 'cfr':
        policy = loads(checkpoint.policy, compression='gzip')
        for agent in env_and_policy.agents:
            agent.policy = policy
    else:
        # Restore the parameters
        policy = env_and_policy.make_policy()
        policy.restore(loads(checkpoint.policy, compression='gzip'))
    return env_and_policy


def load_ppo_agent(best_response):
    # Takes a DB best response object and returns the PPO agent
    db_game = best_response.checkpoint.equilibrium_solver_run.game
    br_agent = make_ppo_agent(best_response.br_player, best_response.config, db_game.load_as_spiel())
    br_agent.restore(loads(best_response.model, compression='gzip'))
    return br_agent

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def setup_logging(filename='auctions.log'):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {"format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"},
                "file": {"format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "console"},
                "file": {"class": "logging.FileHandler", "formatter": "file", "filename": filename},
            },
            "loggers": {"": {"level": "INFO", "handlers": ["console", "file"]}},
        }
    )

def to_ecdf(x):
    if len(x) > 0:
        ecdf = ECDF(x)
        # Need to replace -inf for JSON
        x_vals = ecdf.x
        x_vals[0] = 0
        return zip(x_vals, ecdf.y)
    return []

def is_float(param):
    try:
        float(param)
        return True
    except TypeError:
        return False

def safe_zip(a, b):
    '''Errors if unequal sizes passed in'''
    if len(a) != len(b):
        raise ValueError("Unequal sizes!")
    return zip(a,b)

def get_checkpoint_by_name(experiment_name, run_name, t=None):
    return get_checkpoint(EquilibriumSolverRun.objects.get(name=run_name, experiment__name=experiment_name), t=t)

def get_checkpoint(run, t=None):
    filter_kwargs = dict(equilibrium_solver_run=run)
    if t is None:
        # Get the latest
        t = EquilibriumSolverRunCheckpoint.objects.defer('policy').filter(**filter_kwargs).order_by('-t')[0].t
    filter_kwargs['t'] = t
    equilibrium_solver_run_checkpoint = EquilibriumSolverRunCheckpoint.objects.defer('policy').get(**filter_kwargs)
    return equilibrium_solver_run_checkpoint

def find_best_checkpoint(run, max_t=None):
    ev_df = parse_run(run, max_t)
    best_t = ev_df.query('ApproxNashConv > 0').groupby('t')['ApproxNashConv'].first().idxmin()
    nash_conv_by_t = ev_df.groupby('t')['ApproxNashConv'].first()
    best_checkpoint = get_checkpoint(run, t=best_t)
    return nash_conv_by_t, best_checkpoint, nash_conv_by_t.min()


def convert_pesky_np(d):
    # TODO: This solution is dumb in retrospect: Use a serializer See NpEncoder
    '''Django complains it doesn't know to JSON serialize numpy arrays'''
    if isinstance(d, np.ndarray):
        return convert_pesky_np(d.tolist())
    elif isinstance(d, list):
        return [convert_pesky_np(x) for x in d]
    elif isinstance(d, np.int64):
        return int(d)
    elif isinstance(d, np.float32):
        return float(d)
    elif isinstance(d, dict):
        return {k: convert_pesky_np(v) for k, v in d.items()}
    else:
        return d

    # new_d = dict()
    # for k, v in d.items():
    #     if isinstance(v, list):
    #         new_d[k] = convert_pesky_np(np.array(v))
    #     if isinstance(v, np.ndarray):
    #         new_d[k] = v.tolist()
    #     elif isinstance(v, np.int64):
    #         new_d[k] = int(v)
    #     elif isinstance(v, dict):
    #         new_d[k] = convert_pesky_np(v)
    #     else:
    #         new_d[k] = v
    # return new_d

def add_profiling_flags(parser):
    parser.add_argument('--pprofile', type=util.strtobool, default=0)
    parser.add_argument('--pprofile_file', type=str, default='profile.txt')
    parser.add_argument('--cprofile', type=util.strtobool, default=0)
    parser.add_argument('--cprofile_file', type=str, default='cprofile.txt')
    parser.add_argument('--profile_memory', type=util.strtobool, default=0)

def add_experiment_flags(parser):
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--job_name', type=str, default='auction')
    parser.add_argument('--filename', type=str, default='parameters.json') # Select clock auction game
    parser.add_argument('--game_name', type=str, default='python_clock_auction')

def add_reporting_flags(parser):
    parser.add_argument('--report_freq', type=int, default=50_000)
    parser.add_argument('--eval_every', type=int, default=300_000)
    parser.add_argument('--eval_every_seconds', type=int, default=None)
    parser.add_argument("--eval_every_early", type=int, default=None)
    parser.add_argument("--eval_exactly", nargs="+", default=[], type=int)
    parser.add_argument("--eval_zero", type=util.strtobool, default=1)

def add_dispatching_flags(parser):
    parser.add_argument('--dispatch_br', type=util.strtobool, default=1)
    parser.add_argument('--br_portfolio_path', type=str, default=None)
    parser.add_argument('--br_overrides', type=str, default='', help='These are arguments you want to pass to BR. DO NOT INCLUDE EVAL ARGS HERE')
    parser.add_argument('--eval_overrides', type=str, default='', help="These are arguments you want to pass directly through to evaluate. They ALSO get passed to best respones")
    parser.add_argument('--eval_inline', type=util.strtobool, default=1, help="Eval inline means that we run the eval in the same process as the training")


def add_wandb_flags(parser, default=True):
    parser.add_argument('--use_wandb', type=util.strtobool, default=1 if default else 0) 
    parser.add_argument('--wandb_step_interval', type=int, default=1024, help='Approximate number of steps between wandb logs')
    parser.add_argument('--wandb_note', type=str, default='') 

def add_eval_flags(parser):
    parser.add_argument('--eval_num_samples', type=int, default=EvalDefaults.DEFAULT_NUM_SAMPLES)
    parser.add_argument('--eval_report_freq', type=int, default=EvalDefaults.DEFAULT_REPORT_FREQ)
    parser.add_argument('--eval_num_envs', type=int, default=EvalDefaults.DEFAULT_NUM_ENVS)
    parser.add_argument('--eval_compute_efficiency', type=util.strtobool, default=EvalDefaults.DEFAULT_COMPUTE_EFFICIENCY)
    parser.add_argument('--eval_restrict_to_heuristics', type=util.strtobool, default=EvalDefaults.DEFAULT_RESTRICT_TO_HEURISTICS)



def profile_cmd(cmd, pprofile, pprofile_file, cprofile, cprofile_file):
    if pprofile:
        import pprofile
        profiler = pprofile.Profile()
        with profiler:
            cmd()
        profiler.dump_stats(pprofile_file)
    elif cprofile:
        import cProfile, pstats, io
        profiler = cProfile.Profile()
        profiler.enable()
        cmd()
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open(cprofile_file, 'w+') as f:
            f.write(s.getvalue())
    else:
        cmd()
