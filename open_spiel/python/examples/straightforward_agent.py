from open_spiel.python.rl_agent import AbstractAgent, StepOutput
import numpy as np
from open_spiel.python.examples.ubc_utils import *

class StraightforwardAgent(AbstractAgent):

    def __init__(self, player_id, game):
        self.player_id = player_id
        self.num_actions = game.num_distinct_actions()

    def step(self, time_step, is_evaluation=False):
        if isinstance(time_step, list):
            return [self._act(ts) for ts in time_step]
        else:
            return self._act(time_step)

    def _act(self, time_step):
        if time_step.last():
            return

        legal_actions = time_step.observations["legal_actions"][self.player_id]
        info_dict = time_step.observations["info_dict"][self.player_id]
        profits = info_dict['sor_bundle_prices_history'][-1] # TODO: Why not clock profits? I forget...
        legal_profits = [profits[i] for i in legal_actions]
        
        action = legal_actions[np.argmax(legal_profits)]

        # print("I AM PLAYING ACTION ", action, f"AND MY NAME IS PLAYER {self.player_id}")

        probs = np.zeros(self.num_actions)
        probs[action] = 1.0
        return StepOutput(action=action, probs=probs)