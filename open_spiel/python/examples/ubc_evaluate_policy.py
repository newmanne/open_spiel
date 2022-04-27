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

from open_spiel.python.examples.ubc_utils import fix_seeds, get_player_type, payment_and_allocation, pretty_time, BR_DIR, game_spec, max_num_types, parse_current_round_frame
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator
from open_spiel.python.examples.ubc_cma import efficient_allocation

import numpy as np
import pandas as pd
from absl import logging
import torch
import time
from collections import defaultdict
from dataclasses import dataclass

DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_REPORT_FREQ = 5000
DEFAULT_SEED = 1234
DEFAULT_COMPUTE_EFFICIENCY = False

def run_eval(env_and_model, num_samples, report_freq=DEFAULT_REPORT_FREQ, seed=DEFAULT_SEED, compute_efficiency=DEFAULT_COMPUTE_EFFICIENCY):
    fix_seeds(seed)
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)
    max_types = max_num_types(game_config)

    if compute_efficiency:
      efficiency_df, combo_to_score, efficiency_scorer = efficient_allocation(game, game_config)
      type_combo_to_prob = efficiency_df.set_index('combo')['prob'].to_dict()
      type_combo_to_efficiency = defaultdict(list) # Maps from (type_1, type_2, .. type_n) as integers one for each player into efficiency

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    rewards = defaultdict(list)
    player_types = defaultdict(list)
    allocations = defaultdict(list)
    payments = defaultdict(list)
    efficiencies = []
    episode_lengths = []
    prices = []

    for sample_index in range(num_samples):
      if sample_index % report_freq == 0 and sample_index > 0:
        logging.info(f"----Episode {sample_index} ---")
        for player in range(num_players):
          avg_rewards = pd.Series(rewards[player]).mean()
          logging.info(f"Reward player {player}: {avg_rewards}")

      time_step = env.reset()
      episode_length = 0

      # Get type info
      episode_types = []
      for player_index in range(num_players):
        infostate = time_step.observations['info_state'][player_index]
        player_type = get_player_type(num_players, num_actions, num_products, max_types, infostate)
        player_types[player_index].append(player_type)
        episode_types.append(player_type)
      episode_types = tuple(episode_types)

      episode_rewards = defaultdict(int) # Player ID -> Rewards
      while not time_step.last():
        for i in range(num_players):
          if time_step.rewards is not None:
            episode_rewards[i] += time_step.rewards[i]
        
        episode_length += 1
        player_id = time_step.observations["current_player"]
        agent = agents[player_id]
        agent_output = agent.step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      episode_alloc = []
      final_posted_prices = None # This will get overwritten again and again, but who cares, should always be the same
      for i, agent in enumerate(agents):
        agent.step(time_step, is_evaluation=True)
        episode_rewards[i] += time_step.rewards[i] 
        rewards[i].append(episode_rewards[i])

        # Let's get allocation and pricing information since we're in the last time step
        infostate = time_step.observations['info_state'][i]
        # TODO: These next two function calls could be combined for speed
        payment, allocation = payment_and_allocation(num_players, num_actions, num_products, infostate, max_types)
        final_posted_prices = parse_current_round_frame(num_players, num_actions, num_products, infostate, max_types)['posted_prices']
        allocation = [int(x) for x in allocation]
        payments[i].append(payment)
        episode_alloc.append(allocation)
        allocations[i].append(allocation)

      episode_lengths.append(episode_length)
      if compute_efficiency:
        normalized_efficiency = efficiency_scorer(episode_alloc, episode_types)[1]
        efficiencies.append(normalized_efficiency)
        type_combo_to_efficiency[episode_types].append(normalized_efficiency)
      prices.append(final_posted_prices)
    
    for player in range(num_players):
      logging.info(f"Rewards for {player}")
      logging.info(pd.Series(rewards[player]).describe())
      logging.info(f"-------------------")

    if compute_efficiency:
      overall_efficiency = 0
      type_combo_to_aggregated_efficiency = dict()
      for k, v in type_combo_to_efficiency.items():
        mean_eff = np.mean(v)
        type_combo_to_aggregated_efficiency[k] = mean_eff
        overall_efficiency += mean_eff * type_combo_to_prob[k]

    eval_time = time.time() - alg_start_time
    logging.info(f'Walltime: {pretty_time(eval_time)}')


    checkpoint = {
      'walltime': eval_time,
      'rewards': rewards, # For now, store all the rewards. But maybe we only need some summary stats. Or perhaps a counter is more compressed since few unique values in practice?
      'types': player_types,
      'allocations': allocations,
      'payments': payments,
      'auction_lengths': list((pd.Series(episode_lengths) / num_players)),
      'prices': prices,
    }

    if compute_efficiency:
      checkpoint['efficiencies'] = efficiencies
      checkpoint['efficiency_by_type'] = {','.join(map(str,k)): v for k,v in type_combo_to_aggregated_efficiency.items()} # JSON cant have tuples as keys
      checkpoint['efficiency'] = overall_efficiency

    return checkpoint