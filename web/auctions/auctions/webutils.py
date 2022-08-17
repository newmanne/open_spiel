import logging
from statsmodels.distributions.empirical_distribution import ECDF
from open_spiel.python.examples.ubc_utils import load_game_config
from auctions.models import *
from open_spiel.python.examples.ppo_utils import make_env_and_policy, EnvParams, make_ppo_agent
from open_spiel.python.examples.ubc_plotting_utils import parse_run
import pytz
import datetime
import torch
import pyspiel

NORMALIZATION_DATE = datetime.datetime(2022, 3, 23, 4, 33, 3, 722237, tzinfo=pytz.UTC)

OUTPUT_ROOT = '/shared/outputs'

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

def env_and_policy_for_dry_run(game_db_obj, config):
    game = game_db_obj.load_as_spiel()
    env_and_policy = make_env_and_policy(game, dict(config))
    return env_and_policy

def ppo_db_checkpoint_loader(checkpoint, env_params=None):
    # Create an env_and_policy based on a checkpoint in the database
    env_and_policy = env_and_policy_from_run(checkpoint.equilibrium_solver_run, env_params=env_params)
    # Restore the parameters
    policy = env_and_policy.make_policy()
    policy.restore(pickle.loads(checkpoint.policy))
    return env_and_policy

def load_ppo_agent(best_response):
    # Takes a DB best response object and returns the PPO agent
    db_game = best_response.checkpoint.equilibrium_solver_run.game
    br_agent = make_ppo_agent(best_response.br_player, best_response.config, db_game.load_as_spiel())
    br_agent.restore(pickle.loads(best_response.model))
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
        t = EquilibriumSolverRunCheckpoint.objects.filter(**filter_kwargs).order_by('-t')[0].t
    filter_kwargs['t'] = t
    equilibrium_solver_run_checkpoint = EquilibriumSolverRunCheckpoint.objects.get(**filter_kwargs)
    return equilibrium_solver_run_checkpoint

def find_best_checkpoint(run, max_t=None):
    ev_df = parse_run(run, max_t)
    best_t = ev_df.query('ApproxNashConv > 0').groupby('t')['ApproxNashConv'].first().idxmin()
    nash_conv_by_t = ev_df.groupby('t')['ApproxNashConv'].first()
    best_checkpoint = get_checkpoint(run, t=best_t)
    return nash_conv_by_t, best_checkpoint, nash_conv_by_t.min()
