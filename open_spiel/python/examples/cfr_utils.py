from dataclasses import dataclass, field
from open_spiel.python import rl_environment
from open_spiel.python.examples.env_and_policy import EnvAndPolicy
from open_spiel.python.examples.ubc_utils import *
import numpy as np
from open_spiel.python.pytorch.ppo import PPO
import time
import logging
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.env_decorator import NormalizingEnvDecorator, AuctionStatTrackingDecorator, StateSavingEnvDecorator, PotentialShapingEnvDecorator, TrapEnvDecorator
from typing import Callable, List
from dataclasses import asdict
from open_spiel.python.env_decorator import AuctionStatTrackingDecorator
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states_with_policy
from open_spiel.python.algorithms.outcome_sampling_mccfr import OutcomeSamplingSolver
from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver

logger = logging.getLogger(__name__)

CFR_DEFAULTS = {
}

def read_cfr_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    config = {**CFR_DEFAULTS, **config}  # priority from right to left

    print(config)

    return config

def load_solver(solver_config, game):
    solver_name = solver_config['solver']
    if solver_name == "cfr":
        logger.info("Using CFR solver")
        solver = cfr.CFRSolver(game)
    elif solver_name == "cfrplus":
        logger.info("Using CFR+ solver")
        solver = cfr.CFRPlusSolver(game)
    elif solver_name == "cfrbr":
        logger.info("Using CFR-BR solver")
        solver = cfr.CFRBRSolver(game)
    elif solver_name == "mccfr":
        logger.info("Using MCCFR solver")
        sampling_method = solver_config['sampling_method']

        solver_kwargs = dict()
        if 'linear_averaging' in solver_config:
            solver_kwargs['linear_averaging'] = solver_config['linear_averaging']
        if 'regret_matching_plus' in solver_config:
            solver_kwargs['regret_matching_plus'] = solver_config['regret_matching_plus']
        if 'regret_init' in solver_config:
            solver_kwargs['regret_init'] = solver_config['regret_init']
            solver_kwargs['regret_init_strength'] = solver_config.get('regret_init_strength', 1)

        if sampling_method == "outcome":
            logger.info("Using outcome sampling")

            if 'explore_prob' in solver_config:
                solver_kwargs['explore_prob'] = solver_config['explore_prob']
            if 'tremble_prob' in solver_config:
                solver_kwargs['tremble_prob'] = solver_config['tremble_prob']

            solver = OutcomeSamplingSolver(game, **solver_kwargs)
        else:
            logger.info("Using external sampling")
            solver = ExternalSamplingSolver(game, **solver_kwargs)
    return solver

class CFRAgent:

    def __init__(self, i):
        self.player_id = i
        self.policy = None

    def step(self, time_step, is_evaluation=False):
        if isinstance(time_step, list):
            return [self._act(ts) for ts in time_step]
        else:
            return self._act(time_step)

    def _act(self, time_step):
        if time_step.last():
            return

        state = time_step.observations['state']
        action_probs = self.policy.action_probabilities(state)
        action = fast_choice(list(action_probs.keys()), list(action_probs.values())) # TODO: Give rng

        probs = np.array(list(action_probs.values())) # TODO: is this indexed right? It's good enough to compute entropies...
        return StepOutput(action=action, probs=probs)

def make_cfr_agent(player_id, config, game):
    return CFRAgent(player_id)