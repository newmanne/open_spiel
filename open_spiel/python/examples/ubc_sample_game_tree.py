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

from distutils.log import info
from locale import normalize
from xml.etree.ElementInclude import include

from open_spiel.python.examples.ubc_utils import *

import numpy as np
import pandas as pd
from open_spiel.python.observation import make_observation
import torch
from absl import logging
import time
from collections import defaultdict
from tqdm import tqdm
import re


class GameTreeSampleDefaults:
    DEFAULT_REPORT_FREQ = 5000
    DEFAULT_SEED = 1234

class NodeType:
    ROOT = -1
    INFORMATION_STATE = 0
    ACTION = 1
    FINAL_STATE = 2

def pretty_information_state(info_dict, state, player_id):
    if state.round == 1:
        return str(state.bidders[player_id].bidder)
    else:
        pretty_str = ''
        submitted = info_dict['submitted_demand_history'][-1]
        processed = info_dict['processed_demand_history'][-1]
        aggregate = info_dict['agg_demand_history'][-1]
        if not np.array_equals(submitted, processed):
            pretty_str += f'Processed: {processed}\n'
        pretty_str += f'Aggregate: {aggregate}'
        return pretty_str

def get_input(name, data):
    def hook(model, input, output):
        data[name] = input[0].detach().cpu()
    return hook

def new_tree_node(node_type, str_desc, depth, agent_output=None, env_and_policy=None, time_step=None, agent_to_dqn_embedding=None, parent=None, clusterer=None, include_embeddings=False):
    node = {
        'sample_outcomes': defaultdict(list),
        'children': {},
        'type': node_type,
        'depth': depth
    }
    
    pretty_str = str_desc

    if clusterer is not None:
        if node_type == NodeType.ROOT:
            node['cluster'] = -1
        elif node_type == NodeType.FINAL_STATE:
            node['cluster'] = -2
            node['cluster_from'] = parent['cluster']

    if node_type in [NodeType.INFORMATION_STATE, NodeType.ACTION]:
        player_id = time_step.observations["current_player"]
        agent = env_and_policy.agents[player_id]
        game = env_and_policy.game
        env = env_and_policy.env
        state = env._state
        num_players, num_actions, num_products = game.num_players(), game.num_distinct_actions(), game.auction_params.num_products
        info_dict = time_step.observations["info_dict"][player_id]

        node['player_id'] = player_id
        node['round'] = state.round
        node['feature_vector'] = time_step.observations["info_state"][player_id]

        if node_type == NodeType.INFORMATION_STATE:
            pretty_str = pretty_information_state(info_dict, env._state, player_id)
            node['activity'] = info_dict['activity']
            node['start_of_round_exposure'] = info_dict['sor_exposure'] 
            current_holdings = info_dict['processed_demand_history'][-1]
            agg_demand = info_dict['agg_demand_history'][-1]
            for i in range(num_products):
                letter = num_to_letter(i)
                node[f'Processed {letter}'] = current_holdings[i]
                node[f'Agg Demand {letter}'] = agg_demand[i]
                node[f'Price Increments {letter}'] = info_dict['price_increments'][i]
                
            player_type = state.bidders[player_id]
            node['player_type'] = str(player_type)

            if clusterer is not None:
                node['cluster_from'] = parent['cluster']
            
        elif node_type == NodeType.ACTION:
            action_id = agent_output.action
            
            observer = make_observation(game, params=dict(normalize=False))
            observer.set_from(state, player=player_id)
            unnormalized_info_dict = observer.dict

            profits = np.array(unnormalized_info_dict['clock_profits'])
            profit = profits[action_id]
            legal_actions = env._state.legal_actions() # Only show budget feasible
            max_cp_idx = pd.Series(profits[legal_actions]).idxmax()
            mapped_action_id = legal_actions.index(action_id)

            node['straightforward_clock_profit'] = profit
            node['max_cp'] = mapped_action_id == max_cp_idx

            if include_embeddings:
                # TODO:`    `
                parent['dqn_embedding'] = agent_to_dqn_embedding[player_id].numpy()
                del agent_to_dqn_embedding[player_id] # Clear embedding for safety to make sure we aren't reusing these values and prevent bugs

            bundles = game.auction_params.all_bids
            clock_prices = state.clock_prices[-1]
            for i in range(num_products):
                letter = num_to_letter(i)
                node[f'clock_price {letter}'] = clock_prices[i]

                # Integrate over all actions
                expected_bid = sum([p * q[i] for p, q in zip(agent_output.probs, bundles)])
                node[f'expected_bid {letter}'] = expected_bid
                parent[f'expected_bid {letter}'] = expected_bid
            
            activity = game.auction_params.activity
            parent['expected_activity'] = sum([p * (q @ activity) for p, q in zip(agent_output.probs, bundles)])
            parent['expected_cost'] = sum([p * (q @ clock_prices) for p, q in zip(agent_output.probs, bundles)])

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

# TODO: Rebuild to use SyncVectorEnv for efficiency
def sample_game_tree(env_and_policy, num_samples, report_freq=GameTreeSampleDefaults.DEFAULT_REPORT_FREQ, seed=GameTreeSampleDefaults.DEFAULT_SEED, include_embeddings=False, clusterer=None):
    fix_seeds(seed)

    game, env, agents = env_and_policy.game, env_and_policy.env, env_and_policy.agents
    num_players, num_actions, num_products = game.num_players(), game.num_distinct_actions(), game.auction_params.num_products

    player_to_n_types = dict()
    for player in range(num_players):
        player_to_n_types[player] = len(game.auction_params.player_types[player])

    # # Add hooks
    # # TODO:
    agent_to_dqn_embedding = {}
    # if include_embeddings:
    #     torch_hook_handles = []
    #     agent_to_embedding = {}
    #     for player, agent in enumerate(agents):
    #         if hasattr(agent, '_avg_network'):
    #             handle = agent._avg_network.get_last_layer().register_forward_hook(get_input(player, agent_to_embedding))
    #             torch_hook_handles.append(handle)
    #         if hasattr(agent, '_rl_agent'):
    #             dqn_handle = agent._rl_agent._q_network.get_last_layer().register_forward_hook(get_input(player, agent_to_dqn_embedding))
    #             torch_hook_handles.append(dqn_handle)

    roots = [new_tree_node(NodeType.ROOT, '(root)', 0, clusterer=clusterer, include_embeddings=include_embeddings) for _ in range(num_players)]

    for sample_index in tqdm(range(num_samples)):
        if sample_index % report_freq == 0 and sample_index > 0:
            logging.info(f"----Episode {sample_index} ---")

        time_step = env.reset()
        child_list = [[] for _ in range(num_players)] # Player ID -> List of actions (at own nodes) / infostates (between own nodes)
        current_nodes = [roots[p] for p in range(num_players)] # Player ID -> current position in game tree
        
        while not time_step.last():
            # Find current player
            player_id = time_step.observations["current_player"]
            agent = agents[player_id]

            # Note that we got to this infostate
            infostate_string = env._state.information_state_string()
            new_infostate = infostate_string not in current_nodes[player_id]['children']
            if new_infostate:
                current_nodes[player_id]['children'][infostate_string] = new_tree_node(node_type=NodeType.INFORMATION_STATE, str_desc=infostate_string, depth=current_nodes[player_id]['depth'] + 1, env_and_policy=env_and_policy, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[player_id], clusterer=clusterer, include_embeddings=include_embeddings)
            current_nodes[player_id] = current_nodes[player_id]['children'][infostate_string]

            child_list[player_id].append(infostate_string)
            legal_actions = time_step.observations["legal_actions"][player_id]

            # Choose action
            agent_output = agent.step(time_step, is_evaluation=True)
            if include_embeddings and new_infostate: # Protect against the Caching decorator
                # TODO: This stuff isn't fixed yet. Do we need to separate the critic and the actor layer embeddings? Probably, yes
                if len(legal_actions) > 1:
                    # Protect against information states with a single legal action, which don't call the network
                    current_nodes[player_id]['embedding'] = agent_to_embedding[player_id].numpy()
                    del agent_to_embedding[player_id] # Clear embedding for safety to make sure we aren't reusing these values and prevent bugs

                    if clusterer is not None:
                        current_nodes[player_id]['cluster'] = clusterer(current_nodes[player_id]['embedding'].reshape(1,-1))

            # Note that we took this action
            action_string = env._state.action_to_string(agent_output.action)
            if action_string not in current_nodes[player_id]['children']:
                current_nodes[player_id]['children'][action_string] = new_tree_node(node_type=NodeType.ACTION, str_desc=action_string, depth=current_nodes[player_id]['depth'] + 1, agent_output=agent_output, env_and_policy=env_and_policy, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[player_id], clusterer=clusterer, include_embeddings=include_embeddings)
            current_nodes[player_id] = current_nodes[player_id]['children'][action_string]
            child_list[player_id].append(action_string)

            # Take action
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        stat_dict = env.stats_dict()
        for i, agent in enumerate(agents):
            # Add terminal state
            agent.step(time_step, is_evaluation=True)

            # Get some representation of the final bids
            final_string = str(env._state.get_allocation())
            if final_string not in current_nodes[i]['children']:
                current_nodes[i]['children'][final_string] = new_tree_node(node_type=NodeType.FINAL_STATE, str_desc=final_string, depth=current_nodes[i]['depth'] + 1, env_and_policy=env_and_policy, time_step=time_step, agent_to_dqn_embedding=agent_to_dqn_embedding, parent=current_nodes[i], clusterer=clusterer, include_embeddings=include_embeddings)
            current_nodes[i] = current_nodes[i]['children'][final_string]
            child_list[i].append(final_string)

            # Go back and add outcomes to every intermediate state in the tree
            node = roots[i]
            
            for j in range(len(child_list[i])+1):
                # Add outcomes
                node['sample_outcomes']['profit'].append(stat_dict['raw_rewards'][i][-1])
                node['sample_outcomes']['payment'].append(stat_dict['payments'][i][-1])
                node['sample_outcomes']['allocation'].append(stat_dict['allocations'][i][-1])
                node['sample_outcomes']['rounds'].append(stat_dict['auction_lengths'][-1])
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
    