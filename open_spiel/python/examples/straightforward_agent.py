from open_spiel.python.rl_agent import AbstractAgent, StepOutput
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value
import numpy as np
import pandas as pd
from absl import logging
from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.observation import make_observation

# TODO: Currently breaks if you don't use TakeOnlyActionDecorator to deal w/ zeros in objective
class StraightforwardAgent(AbstractAgent):

    def __init__(self, player_id, game):
        self.player_id = player_id
        self.num_actions = game.num_distinct_actions()
        self.obs = make_observation(game)

    def step(self, time_step, is_evaluation=False):
        if time_step.last():
            return

        legal_actions = time_step.observations["legal_actions"][self.player_id]
        info_dict = time_step.observations["info_dict"][self.player_id]
        profits = info_dict['sor_profits']
        legal_profits = [profits[i] for i in legal_actions]
        action = legal_actions[np.argmax(legal_profits)]

        probs = np.zeros(self.num_actions)
        probs[action] = 1.0
        return StepOutput(action=action, probs=probs)