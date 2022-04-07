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
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator

import numpy as np
import pandas as pd
import torch
from absl import app, logging
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

def get_input(name, data):
    def hook(model, input, output):
        data[name] = input[0].detach().cpu()
    return hook

def new_tree_node(node_type, str_desc, depth, agent_output=None, env_and_model=None, time_step=None, agent_to_dqn_embedding=None, parent=None, clusterer=None):
    node = {'sample_outcomes': defaultdict(list), 'children': {}, 'type': node_type, 'depth': depth} # 'pretty_name': pretty_str}
    pretty_str = str_desc

    if node_type == NodeType.ROOT:
        if clusterer is not None:
            node['cluster'] = -1

    if node_type == NodeType.FINAL_STATE:
        if clusterer is not None:
            node['cluster'] = -2
            node['cluster_from'] = parent['cluster']

    if node_type in [NodeType.INFORMATION_STATE, NodeType.ACTION]:
        player_id = time_step.observations["current_player"]
        information_state_tensor = time_step.observations["info_state"][player_id]
        game = env_and_model.game
        agent = env_and_model.agents[player_id]
        state = env_and_model.env._state
        game_config = env_and_model.game_config

        num_players, num_actions, num_products = game_spec(game, game_config)
        max_types = max_num_types(game_config)

        clock_prices_index = clock_price_index(num_players, num_actions)
        clock_prices = np.array(information_state_tensor[clock_prices_index:clock_prices_index + num_products])

        node['player_id'] = player_id
        node['round'] = information_state_tensor[round_index(num_players)]
        node['feature_vector'] = information_state_tensor[turn_based_size(num_players):turn_based_size(num_players) + handcrafted_size(num_actions, num_products)]

        if node_type == NodeType.INFORMATION_STATE:
            pretty_str = pretty_information_state(str_desc, depth)
            node['activity'] = information_state_tensor[activity_index(num_players, num_actions)]
            node['start_of_round_exposure'] = information_state_tensor[sor_exposure_index(num_players, num_actions)]
            current_round_frame_parsed = parse_current_round_frame(num_players, num_actions, num_products, information_state_tensor, max_types)
            current_holdings = current_round_frame_parsed['allocation']
            agg_demand = current_round_frame_parsed['agg_demand']
            for i in range(num_products):
                letter = num_to_letter(i)
                node[f'Processed {letter}'] = current_holdings[i]
                node[f'Agg Demand {letter}'] = agg_demand[i]
                node[f'Price Increments {letter}'] = num_increments(clock_prices[i], game_config['increment'], game_config['opening_price'][i])
                
            player_type_index = get_player_type(num_players, num_actions, num_products, max_types, information_state_tensor) 
            player_type = type_from_index(game_config, player_id, player_type_index)
            node['player_type'] = f'b{player_type["budget"]}v{",".join(map(str, player_type["value"]))}'

            if clusterer is not None:
                node['cluster_from'] = parent['cluster']
            
        elif node_type == NodeType.ACTION:
            action_id = agent_output.action

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
                # Trigger the DQN
                # TODO: Calling this for each action and not storing the results on the infostate level may lead to excessive calls
                q_values = check_on_q_values(rl_agent, game, time_step=time_step, return_raw_q_values=True)
                node['q_value'] = q_values[action_id]
                # TODO: I want this on the parent...
                parent['dqn_embedding'] = agent_to_dqn_embedding[player_id].numpy()
                del agent_to_dqn_embedding[player_id] # Clear embedding for safety to make sure we aren't reusing these values and prevent bugs

            bundles = action_to_bundles(env_and_model.game_config['licenses'])

            for i in range(num_products):
                letter = num_to_letter(i)
                node[f'clock_price {letter}'] = clock_prices[i]

                # Integrate over all actions
                expected_bid = sum([p * q[i] for p, q in zip(agent_output.probs, bundles.values())])
                node[f'expected_bid {letter}'] = expected_bid
                parent[f'expected_bid {letter}'] = expected_bid
            
            activity = np.array(env_and_model.game_config['activity'])
            parent['expected_activity'] = sum([p * (q @ activity) for p, q in zip(agent_output.probs, bundles.values())])
            parent['expected_cost'] = sum([p * (q @ clock_prices) for p, q in zip(agent_output.probs, bundles.values())])

            if clusterer is not None:
                node['cluster'] = parent['cluster']

    node['pretty_str'] = pretty_str
    return node

def aggregate_results(node, num_samples):
    # Count number of observations

    node['num_plays'] = len(node['sample_outcomes']['profit'])
    node['pct_plays'] = (node['num_plays'] / num_samples) * 100
    # Add means in place
    # TODO: save other stats?
    for k in node['sample_outcomes']:
        if k == 'allocation':
            avgs = np.mean(node['sample_outcomes'][k], axis=0)
            for i in range(len(node['sample_outcomes'][k][0])):
                letter = num_to_letter(i)
                node[f'Avg Alloc {letter}'] = avgs[i]
        else:
            node[f'avg_{k}'] = np.mean(node['sample_outcomes'][k], axis=0)
    del node['sample_outcomes']

    for child in node['children']:
        aggregate_results(node['children'][child], num_samples)

def sample_game_tree(env_and_model, num_samples, report_freq=DEFAULT_REPORT_FREQ, seed=DEFAULT_SEED, include_embeddings=False, clusterer=None):
    fix_seeds(seed)

    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)
    max_types = max_num_types(game_config)

    player_to_n_types = dict()
    for player in range(num_players):
        player_to_n_types[player] = len(game_config['players'][player]['type'])

    if include_embeddings:
        for agent in agents:
        #### WARNING: If you use include_embeddings, your agents internal caches (this is different from the caching agent decorator) will cause trouble if you call this function multiple times w/ the same env_and_model. Therefore, we clear the caches here
            agent._clear_cache()

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 
    
    # Add hooks
    agent_to_dqn_embedding = {}
    if include_embeddings:
        torch_hook_handles = []
        agent_to_embedding = {}
        for player, agent in enumerate(agents):
            handle = agent._avg_network.get_last_layer().register_forward_hook(get_input(player, agent_to_embedding))
            dqn_handle = agent._rl_agent._q_network.get_last_layer().register_forward_hook(get_input(player, agent_to_dqn_embedding))
            torch_hook_handles.append(handle)
            torch_hook_handles.append(dqn_handle)

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    roots = [new_tree_node(NodeType.ROOT, '(root)', 0, clusterer=clusterer) for player in range(num_players)]

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
            new_infostate = infostate_string not in current_nodes[player_id]['children']
            if new_infostate:
                current_nodes[player_id]['children'][infostate_string] = new_tree_node(node_type=NodeType.INFORMATION_STATE, str_desc=infostate_string, depth=current_nodes[player_id]['depth'] + 1, env_and_model=env_and_model, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[player_id], clusterer=clusterer)
            current_nodes[player_id] = current_nodes[player_id]['children'][infostate_string]
            child_list[player_id].append(infostate_string)
            legal_actions = time_step.observations["legal_actions"][player_id]

            # Choose action
            agent_output = agent.step(time_step, is_evaluation=True)
            if include_embeddings and new_infostate: # Protect against the Caching decorator
                if len(legal_actions) > 1:
                    # Protect against information states with a single legal action, which don't call the network
                    current_nodes[player_id]['embedding'] = agent_to_embedding[player_id].numpy()
                    del agent_to_embedding[player_id] # Clear embedding for safety to make sure we aren't reusing these values and prevent bugs

                    if clusterer is not None:
                        current_nodes[player_id]['cluster'] = clusterer(current_nodes[player_id]['embedding'].reshape(1,-1))

            # Note that we took this action
            action_string = env._state.action_to_string(agent_output.action)
            if action_string not in current_nodes[player_id]['children']:
                current_nodes[player_id]['children'][action_string] = new_tree_node(node_type=NodeType.ACTION, str_desc=action_string, depth=current_nodes[player_id]['depth'] + 1, agent_output=agent_output, env_and_model=env_and_model, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[player_id], clusterer=clusterer)
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
                current_nodes[i]['children'][final_string] = new_tree_node(node_type=NodeType.FINAL_STATE, str_desc=final_string, depth=current_nodes[i]['depth'] + 1, env_and_model=env_and_model, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[i], clusterer=clusterer)
            current_nodes[i] = current_nodes[i]['children'][final_string]
            child_list[i].append(final_string)

            # Compute outcomes
            infostate = time_step.observations['info_state'][i]
            payment, allocation = payment_and_allocation(num_players, num_actions, num_products, infostate, max_types)

            # Go back and add outcomes to every intermediate state in the tree
            node = roots[i]
            for j in range(len(child_list[i])+1):
                # Add outcomes
                node['sample_outcomes']['profit'].append(episode_rewards[i])
                node['sample_outcomes']['payment'].append(payment)
                node['sample_outcomes']['allocation'].append(allocation)
                node['sample_outcomes']['rounds'].append(episode_length / num_players)
                for opponent in players_not_me(i, num_players):
                    opponent_type_id = env._state.history()[opponent] # Super implementation specific, where this is the ith chance outcome. Better would be to parse str(state) or expose a metrics() function from state to pyspiel
                    for type_index in range(player_to_n_types[opponent]):
                        node['sample_outcomes'][f'p{opponent}_type_{type_index}'].append(1 if type_index == opponent_type_id else 0)

                # Move to next node
                if j < len(child_list[i]):
                    node = node['children'][child_list[i][j]]

    # Remove torch handles if applied to help reuse of function
    if include_embeddings:
        for handle in torch_hook_handles:
            handle.remove()

    # Take mean outcome at each node
    for i in range(num_players):
        aggregate_results(roots[i], num_samples) 

    return roots           

def flatten_tree(node):
    records = []
    
    if node['type'] == NodeType.INFORMATION_STATE:
        tree_features = dict(node)
        del tree_features['children']
        records.append(tree_features)
    
    for child in node['children']:
        records += flatten_tree(node['children'][child])
    
    return records
    
def flatten_trees(trees):
    records = []
    for tree in trees:
        records += flatten_tree(tree)
    df = pd.DataFrame.from_records(records)
    df['player_type'] = df['player_type'].astype('category')
    return df
    
if __name__ == "__main__":
    app.run(main)
