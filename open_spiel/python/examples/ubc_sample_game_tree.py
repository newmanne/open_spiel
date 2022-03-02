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

from open_spiel.python import rl_environment, policy
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, fix_seeds, get_player_type, current_round, round_frame, payment_and_allocation, pretty_time, BR_DIR, game_spec
from open_spiel.python.examples.ubc_nfsp_example import lookup_model_and_args
from open_spiel.python.examples.ubc_br import make_dqn_agent
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator, TakeSingleActionDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from open_spiel.python.examples.legacy_file_classes import policy_from_checkpoint

import pyspiel
import numpy as np
import pandas as pd
import absl
import argparse
from absl import app, logging, flags
import torch
import yaml
import sys
import time
import pickle
import os
import json
import shutil
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

DEFAULT_REPORT_FREQ = 5000
DEFAULT_SEED = 1234


def new_tree_node():
    return {'sample_outcomes': defaultdict(list), 'children': {}}

def aggregate_results(node):
    # Count number of observations
    node['num_plays'] = len(node['sample_outcomes']['profit'])
    # Add means in place
    # TODO: handle allocations differently? 
    # TODO: save other stats?
    for k in node['sample_outcomes']:
        node[f'avg_{k}'] = np.round(np.mean(node['sample_outcomes'][k], axis=0), 2)
    del node['sample_outcomes']

    for child in node['children']:
        aggregate_results(node['children'][child])

def sample_game_tree(env_and_model, num_samples, report_freq=DEFAULT_REPORT_FREQ, seed=DEFAULT_SEED):
    fix_seeds(seed)
    
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    roots = [new_tree_node() for player in range(num_players)]

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
                current_nodes[player_id]['children'][infostate_string] = new_tree_node()
            current_nodes[player_id] = current_nodes[player_id]['children'][infostate_string]
            child_list[player_id].append(infostate_string)

            # Choose action
            agent_output = agent.step(time_step, is_evaluation=True)

            # Note that we took this action
            action_string = env._state.action_to_string(agent_output.action)
            if action_string not in current_nodes[player_id]['children']:
                current_nodes[player_id]['children'][action_string] = new_tree_node()
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
                current_nodes[i]['children'][final_string] = new_tree_node()
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
                node['sample_outcomes']['rounds'].append(episode_length / 2),

                # Move to next node
                if j < len(child_list[i]):
                    node = node['children'][child_list[i][j]]

    # Take mean outcome at each node
    for i in range(num_players):
        aggregate_results(roots[i]) 

    return roots            
    
if __name__ == "__main__":
    app.run(main)
