from open_spiel.python.rl_agent import AbstractAgent, StepOutput
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value
import numpy as np
import pandas as pd
from absl import logging
from open_spiel.python.examples.ubc_utils import *

# TODO: Currently breaks if you don't use TakeOnlyActionDecorator to deal w/ zeros in objective
class StraightforwardAgent(AbstractAgent):

    def __init__(self, player_id, game_config, num_actions):
        self.player_id = player_id
        self.game_config = game_config
        self.supply = np.array(game_config['licenses'])
        self.increment = game_config['increment']
        self.num_products = len(self.supply)
        self.num_actions = num_actions
        self.num_players = len(game_config['players'])
        self.action_to_bundle = action_to_bundles(game_config['licenses'])

    def step(self, time_step, is_evaluation=False):
        if time_step.last():
            return


        information_state_tensor = np.array(time_step.observations["info_state"][self.player_id])
        
        # TODO: Move this into ubc_utils (and ideally for the RNN too) so all manipulations of the tensor are centralized if it has to change

        # Remove encoding from turn based simultaneous game
        information_state_tensor = information_state_tensor[self.num_players * 2:]

        # Skip our own player encoding
        information_state_tensor = information_state_tensor[self.num_players:]

        # Grab the budget
        budget = information_state_tensor[0]

        # Grab the values
        values = information_state_tensor[1:self.num_products + 1]

        # Tensor looks like: Submitted / Processed / Agg / Posted Price
        posted_prices = information_state_tensor[information_state_tensor.nonzero()][-self.num_products:] # Hacky
        current_clock_prices = np.array(posted_prices) * self.increment # The encoding doesn't have clock prices! Need to rederive them (multiply by self.increment)

        legal_actions = time_step.observations["legal_actions"][self.player_id]

        # TODO: This isn't really a MIP - you just literally pick the best bundle. It's a max function. Do you need MIP power later, or should you rewrite this?

        # Convert legal actions into packages and only write the MIP over these packages => Budget + Eligibility taken care of, not MIP constraints
        legal_bundles = []
        expected_profits = []
        # Step 1: Get all of the bundles from legal actions. How much do they each cost? How much do I value each of them?
        for var_id, action in enumerate(legal_actions):
            bundle = self.action_to_bundle[action]
            bundle_value = values @ bundle
            bundle_price = current_clock_prices @ bundle
            bundle_profit = bundle_value - bundle_price
            legal_bundles.append(bundle)
            expected_profits.append(bundle_profit)

        # Extract current clock prices
        # Extract my own values
        # Extract my budget
        # Place the bid that maximizes my profit at clock prices subject to budget (TODO: This is not sophisticated enough)

        feasible_result = True
        problem = LpProblem(f"{self.player_id}_Bid", LpMaximize)

        bundle_variables = LpVariable.dicts("X", np.arange(len(expected_profits)), cat=LpBinary)

        # OBJECTIVE
        problem += lpDot(expected_profits, bundle_variables.values())

        # Constraint: Only 1 bundle
        problem += lpSum(bundle_variables.values()) == 1

        try: 
            problem.writeLP(f'straightforward.lp')
            obj = pulp_solve(problem, save_if_failed=True)
            for var_id in range(len(bundle_variables)):
                if value(bundle_variables[var_id]) > .99: # Rounding stupidness
                    action = var_id
                    break # Only 1 feasible bundle
        except ValueError as e:
            # if MIP is infeasible, drop out - TODO: Should this ever happen?
            feasible_result = False
            logging.warning(f'Failed to solve MIP; dropping out')
            action = 0
        
        probs = np.zeros(self.num_actions)
        probs[action] = 1.0
        return StepOutput(action=action, probs=probs)