import logging
from statsmodels.distributions.empirical_distribution import ECDF
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game
from auctions.models import *
from open_spiel.python.examples.ubc_nfsp_example import setup
from open_spiel.python.examples.ubc_br import make_dqn_agent
from open_spiel.python.examples.ubc_plotting_utils import parse_run

OUTPUT_ROOT = '/shared/outputs'

def load_dqn_agent(best_response):
    # Takes a DB best response object and returns the DQN agent
    db_game = best_response.checkpoint.equilibrium_solver_run.game
    br_agent = make_dqn_agent(best_response.br_player, best_response.config, load_game(db_game), db_game.config)
    br_agent._q_network.load_state_dict(pickle.loads(best_response.model))
    return br_agent

def env_and_model_from_run(run):
    # Retrieve the game
    game_db_obj = run.game
    game = load_game(game_db_obj)
    game_config = game_db_obj.config

    # Get the NFSP config
    config = dict(run.config)

    # Create env_and_model
    env_and_model = setup(game, game_config, config)
    return env_and_model

def db_checkpoint_loader(checkpoint):
    # Create an env_and_model based on an NFSP checkpoint in the database
    env_and_model = env_and_model_from_run(checkpoint.equilibrium_solver_run)

    # Restore the parameters
    nfsp_policies = env_and_model.nfsp_policies
    nfsp_policies.restore(pickle.loads(checkpoint.policy))
    return env_and_model

def load_game(game): # Takes in Django Game object
    return smart_load_sequential_game('clock_auction', dict(filename=game.name))


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
    best_t = ev_df.groupby('t')['ApproxNashConv'].first().idxmin()
    approx_nash_conv = ev_df.groupby('t')['ApproxNashConv'].first().min()
    best_checkpoint = get_checkpoint(run, t=best_t)
    return best_checkpoint, approx_nash_conv