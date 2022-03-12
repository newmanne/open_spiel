    # Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.examples.ubc_nfsp_example import lookup_model_and_args
from open_spiel.python.examples.ubc_br import make_dqn_agent
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator, TakeSingleActionDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from open_spiel.python.examples.legacy_file_classes import policy_from_checkpoint

import numpy as np
import pandas as pd
from absl import app, logging, flags
import torch
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import re

DEFAULT_REPORT_FREQ = 5000
DEFAULT_SEED = 1234

class NodeType:
    ROOT = -1
    INFORMATION_STATE = 0
    ACTION = 1
    FINAL_STATE = 2

price_pattern = re.compile(r'^.*(Price:.*)$', flags=re.MULTILINE)
allocation_pattern = re.compile(r'.*(Final bids:.*)$.*', flags=re.MULTILINE)
round_pattern = re.compile(r'.*(Round:.*)$.*', flags=re.MULTILINE)
value_pattern = re.compile(r'v(.+)b.*')
budget_pattern = re.compile(r'.*b(.+)$')
big_budget_pattern = re.compile(r'Budget: \d+', flags=re.MULTILINE)

def pretty_information_state(infostate_string, depth):
    if depth == 1:
        v_idx = infostate_string.index('Values')
        return infostate_string[v_idx:]
    else:
        # TODO: Clean this up.
        # Rows are submitted, processed, aggregate. I don't care about submitted because it's in the history table. I only care about the last | because the rest is in history
        """
        2, 3, 0 | 3, 1, 0 
        2, 3, 0 | 3, 1, 0
        4, 6, 3 | 4, 4, 3
        """
        res = re.search(big_budget_pattern, infostate_string)
        data_lines = infostate_string[res.end():]
        # ['', '3, 3, 3 | 1, 2, 0', '3, 3, 3 | 3, 2, 0', '3, 6, 6 | 3, 4, 3']
        _, submitted, processed, aggregate = data_lines.split('\n')
        submitted_final = submitted.split('|')[-1].strip()
        processed_final = processed.split('|')[-1].strip()
        aggregate_final = aggregate.split('|')[-1].strip()
        pretty_str = ''
        if submitted_final != processed_final:
            # Only show processed when it's different from submitted. Maybe we should highlight this in red?
            pretty_str += f'Processed: {processed_final}\n'
        pretty_str += f'Aggregate: {aggregate_final}'
        return pretty_str

def new_tree_node(node_type, str_desc, depth, action_id=None, env_and_model=None, time_step=None):
    node = {'sample_outcomes': defaultdict(list), 'children': {}, 'type': node_type, 'depth': depth} # 'pretty_name': pretty_str}
    if node_type == NodeType.INFORMATION_STATE:
        pretty_str = pretty_information_state(str_desc, depth)
        # TODO: Add the aggregate demand history so we can show a separate table?
        # node['agg_demand'] = 
    elif node_type == NodeType.ACTION:
        pretty_str = str_desc

        player_id = time_step.observations["current_player"]
        agent = env_and_model.agents[player_id]
        state = env_and_model.env._state
        game = env_and_model.game
        num_players, num_actions, num_products = game_spec(game, env_and_model.game_config)

        information_state_tensor = time_step.observations["info_state"][player_id]
        cpi = clock_profit_index(num_players, num_actions)
        profits = np.array(information_state_tensor[cpi:cpi + num_actions])
        profit = profits[action_id]
        legal_actions = state.legal_actions() # Only show budget feasible
        max_cp_idx = pd.Series(profits[legal_actions]).idxmax()
        mapped_action_id = legal_actions.index(action_id)
        node['straightforward_clock_profit'] = profit
        node['max_cp'] = mapped_action_id == max_cp_idx
        rl_agent = getattr(agent, '_rl_agent', None)
        if rl_agent is None: # Maybe it already is a DQN? But need to guard against it being a Straightforward agent. And also could be wrapped in a decorator..
            if getattr(agent, '_target_q_network', None):
                rl_agent = agent
        if rl_agent is not None:
            q_values = check_on_q_values(rl_agent, game, time_step=time_step, return_raw_q_values=True)
            node['q_value'] = q_values[action_id]
        
        clock_prices_index = clock_price_index(num_players, num_actions)
        clock_prices = np.array(information_state_tensor[clock_prices_index:clock_prices_index + num_actions])
        for i in range(num_products):
            letter = chr(ord('@')+i+1)
            node[f'clock_price {letter}'] = clock_prices[i]

    else:
        pretty_str = str_desc
    node['pretty_str'] = pretty_str
    return node

def aggregate_results(node, num_samples):
    # Count number of observations

    # TODO: node['normalized_num_plays']?

    node['num_plays'] = len(node['sample_outcomes']['profit'])
    node['pct_plays'] = (node['num_plays'] / num_samples) * 100
    # Add means in place
    # TODO: handle allocations differently? 
    # TODO: save other stats?
    for k in node['sample_outcomes']:
        if k == 'allocation':
            avgs = np.mean(node['sample_outcomes'][k], axis=0)
            for i in range(len(node['sample_outcomes'][k][0])):
                letter = chr(ord('@')+i+1)
                node[f'Avg Alloc {letter}'] = avgs[i]
        else:
            node[f'avg_{k}'] = np.mean(node['sample_outcomes'][k], axis=0)
    del node['sample_outcomes']

    for child in node['children']:
        aggregate_results(node['children'][child], num_samples)

def sample_game_tree(env_and_model, num_samples, report_freq=DEFAULT_REPORT_FREQ, seed=DEFAULT_SEED):
    fix_seeds(seed)
    
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    roots = [new_tree_node(NodeType.ROOT, '(root)', 0) for player in range(num_players)]

    # rewards = defaultdict(list)
    # player_types = defaultdict(list)
    # allocations = defaultdict(list)
    # payments = defaultdict(list)
    # episode_lengths = []

    for sample_index in tqdm(range(num_samples)):
        if sample_index % report_freq == 0 and sample_index > 0:
            logging.info(f"----Episode {sample_index} ---")

        time_step = env.reset()
        episode_length = 0
        episode_rewards = defaultdict(int) # Player ID -> Rewards

        child_list = [[] for _ in range(num_players)] # Player ID -> List of actions (at own nodes) / infostates (between own nodes)
        current_nodes = [roots[p] for p in range(num_players)] # Player ID -> current position in game tree
        
        while not time_step.last():
            for i in range(num_players):
                if time_step.rewards is not None:
                    episode_rewards[i] += time_step.rewards[i]
        
            episode_length += 1

            # Find current player
            player_id = time_step.observations["current_player"]
            agent = agents[player_id]

            # Note that we got to this infostate
            infostate_string = env._state.information_state_string()
            if infostate_string not in current_nodes[player_id]['children']:
                current_nodes[player_id]['children'][infostate_string] = new_tree_node(node_type=NodeType.INFORMATION_STATE, str_desc=infostate_string, depth=current_nodes[player_id]['depth'] + 1, env_and_model=env_and_model, time_step=time_step)
            current_nodes[player_id] = current_nodes[player_id]['children'][infostate_string]
            child_list[player_id].append(infostate_string)

            # Choose action
            agent_output = agent.step(time_step, is_evaluation=True)

            # Note that we took this action
            action_string = env._state.action_to_string(agent_output.action)
            if action_string not in current_nodes[player_id]['children']:
                current_nodes[player_id]['children'][action_string] = new_tree_node(node_type=NodeType.ACTION, str_desc=action_string, depth=current_nodes[player_id]['depth'] + 1, action_id=agent_output.action, env_and_model=env_and_model, time_step=time_step)
            current_nodes[player_id] = current_nodes[player_id]['children'][action_string]
            child_list[player_id].append(action_string)

            # Take action
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        for i, agent in enumerate(agents):
            # Add terminal state
            agent.step(time_step, is_evaluation=True)
            episode_rewards[i] += time_step.rewards[i] 

            # Get some representation of the final bids
            final_string = str(env._state).split('\n')[-1]
            if final_string not in current_nodes[i]['children']:
                current_nodes[i]['children'][final_string] = new_tree_node(node_type=NodeType.FINAL_STATE, str_desc=final_string, depth=current_nodes[i]['depth'] + 1, env_and_model=env_and_model, time_step=time_step)
            current_nodes[i] = current_nodes[i]['children'][final_string]
            child_list[i].append(final_string)

            # Compute outcomes
            infostate = time_step.observations['info_state'][i]
            payment, allocation = payment_and_allocation(num_players, num_actions, num_products, infostate)

            # Go back and add outcomes to every intermediate state in the tree
            node = roots[i]
            for j in range(len(child_list[i])+1):
                # Add outcomes
                node['sample_outcomes']['profit'].append(episode_rewards[i]),
                node['sample_outcomes']['payment'].append(payment),
                node['sample_outcomes']['allocation'].append(allocation),
                node['sample_outcomes']['rounds'].append(episode_length / num_players),

                # Move to next node
                if j < len(child_list[i]):
                    node = node['children'][child_list[i][j]]

    # Take mean outcome at each node
    for i in range(num_players):
        aggregate_results(roots[i], num_samples) 

    return roots            
    
if __name__ == "__main__":
    app.run(main)
