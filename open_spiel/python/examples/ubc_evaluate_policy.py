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
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, fix_seeds, get_player_type, current_round, round_frame, payment_and_allocation
from open_spiel.python.examples.ubc_nfsp_example import policy_from_checkpoint, lookup_model_and_args
from open_spiel.python.examples.ubc_br import BR_DIR, make_dqn_agent
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator, TakeSingleActionDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent

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

EVAL_DIR = 'evaluations'

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_latest')
    parser.add_argument('--num_samples', type=int, default=100_000)
    parser.add_argument('--report_freq', type=int, default=5000)
    parser.add_argument('--br_name', type=str, default=None)
    parser.add_argument('--straightforward_player', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args(argv[1:])  # Let argparse parse the rest of flags.

    alg_start_time = time.time()
    experiment_dir = args.experiment_dir
    checkpoint_name = args.checkpoint
    num_samples = args.num_samples
    report_freq = args.report_freq
    br_name = args.br_name
    straightforward_player = args.straightforward_player
    seed = args.seed
    output_name = args.output

    fix_seeds(seed) 

    if straightforward_player is not None and br_name is not None:
      raise ValueError("Only one of straightforward_player and br_name can be set")

    name = checkpoint_name
    if br_name:
      name += f'_{br_name}'
    elif straightforward_player is not None:
      name += f'_straightforward_{straightforward_player}'

    checkpoint_dir = os.path.join(experiment_dir, EVAL_DIR)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logging.get_absl_handler().use_absl_log_file(f'evaluate_policy_{name}', checkpoint_dir) 
    logging.set_verbosity(logging.INFO)

    env_and_model = policy_from_checkpoint(experiment_dir, checkpoint_suffix=checkpoint_name, env_seed=seed)
    game, policy, env, trained_agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config

    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    num_products = len(game_config['licenses'])

    br_agent_id = None
    agents = trained_agents


    if straightforward_player is not None: # Replace one agent with Straightforward Bidding
      agents[straightforward_player] = TakeSingleActionDecorator(StraightforwardAgent(straightforward_player, game_config, game.num_distinct_actions()), game.num_distinct_actions())
    elif br_name is not None:
      logging.info(f"Reading from agent {br_name}")
      with open(f'{experiment_dir}/{BR_DIR}/{br_name}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
      br_agent_id = checkpoint['br_player']
      config = checkpoint['config']
      br_agent = make_dqn_agent(br_agent_id, config, env, game, game_config)
      br_agent._q_network.load_state_dict(checkpoint['agent'])
      agents[br_agent_id] = br_agent # Replace with our agent
    else:
      logging.info("No best reponders provided. Just evaluating the policy")

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    rewards = defaultdict(list)
    player_types = defaultdict(list)
    allocations = defaultdict(list)
    payments = defaultdict(list)

    for sample_index in range(num_samples):
      if sample_index % report_freq == 0 and sample_index > 0:
        logging.info(f"----Episode {sample_index} ---")
        for player in range(num_players):
          avg_rewards = pd.Series(rewards[player]).mean()
          logging.info(f"Reward player {player}: {avg_rewards}")

      time_step = env.reset()

      # Get type info. Another way to do this might be to instrument the chance sampler...
      for player_index in range(num_players):
        infostate = time_step.observations['info_state'][player_index]
        player_type = get_player_type(num_players, num_actions, num_products, infostate)
        player_types[player_index].append(tuple(player_type))

      episode_rewards = defaultdict(int) # Player ID -> Rewards
      while not time_step.last():
        for i in range(num_players):
          if time_step.rewards is not None:
            episode_rewards[i] += time_step.rewards[i]
        player_id = time_step.observations["current_player"]
        agent = agents[player_id]
        agent_output = agent.step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      for i, agent in enumerate(agents):
        agent.step(time_step, is_evaluation=True)
        episode_rewards[i] += time_step.rewards[i] 
        rewards[i].append(episode_rewards[i])

        # Let's get allocation and pricing information
        infostate = time_step.observations['info_state'][i]
        payment, allocation = payment_and_allocation(num_players, num_actions, num_products, infostate)
        payments[i].append(payment)
        allocations[i].append(allocation)
    
    for player in range(num_players):
      logging.info(f"Rewards for {player}")
      logging.info(pd.Series(rewards[player]).describe())
      logging.info(f"-------------------")

    checkpoint = {
      'walltime': time.time() - alg_start_time,
      'rewards': rewards, # For now, store all the rewards. But maybe we only need some summary stats
      'types': player_types,
      'allocations': allocations,
      'payments': payments,
      'br_agent': br_agent_id,
      'straightforward_agent': straightforward_player
    }

    if output_name is None:
      output_name = f'rewards_{name}'

    checkpoint_path = os.path.join(checkpoint_dir, f'{output_name}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    logging.info('All done. Goodbye!')

    
if __name__ == "__main__":
    app.run(main)
