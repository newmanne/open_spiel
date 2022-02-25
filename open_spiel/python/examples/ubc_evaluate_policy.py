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

def add_argparse_args(parser):
    parser.add_argument('--num_samples', type=int, default=100_000)
    parser.add_argument('--report_freq', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--br_name', type=str, default=None)

def main(argv):
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)

    # File system only arguments
    parser.add_argument('--experiment_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_latest')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--straightforward_player', type=int, default=None) 

    args = parser.parse_args(argv[1:])  # Let argparse parse the rest of flags.

    # Fileysystem-specific args
    experiment_dir = args.experiment_dir
    checkpoint_name = args.checkpoint
    br_name = args.br_name
    output_name = args.output
    straightforward_player = args.straightforward_player

    # General args
    num_samples = args.num_samples
    report_freq = args.report_freq
    seed = args.seed

    logging.get_absl_handler().use_absl_log_file(f'evaluate_policy_{name}', checkpoint_dir) 
    logging.set_verbosity(logging.INFO)
  
    fix_seeds(seed)
    
    env_and_model = policy_from_checkpoint(experiment_dir, checkpoint_suffix=checkpoint_name)
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    br_agent_id = None

    if br_name is None:
      logging.info("No best reponders provided. Just evaluating the policy")
    else:
      if br_name == 'straightforward': # Replace one agent with Straightforward Bidding
        agents[straightforward_player] = TakeSingleActionDecorator(StraightforwardAgent(straightforward_player, game_config, game.num_distinct_actions()), game.num_distinct_actions())
      else:
        br_agent = dqn_agent_from_checkpoint(experiment_dir, checkpoint_name, br_name)
        agents[br_agent.player_id] = br_agent

    eval_output = run_eval(env_and_model, num_samples, report_freq, seed)

    # Save result
    eval_output['br_agent'] = br_agent_id
    eval_output['br_name'] = br_name

    if output_name is None:
      name = checkpoint_name
      if br_name:
        name += f'_{br_name}'
      output_name = f'rewards_{name}'

    checkpoint_path = os.path.join(checkpoint_dir, f'{output_name}.pkl')
    with open(checkpoint_path, 'wb') as f:
      pickle.dump(eval_output, f)

    logging.info('All done. Goodbye!')


def run_eval(env_and_model, num_samples, report_freq, seed):
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    # Apply cache
    agents = [CachingAgentDecorator(agent) for agent in agents] 

    # EVALUATION PHASE
    logging.info(f"Evaluation phase: {num_samples} episodes")
    alg_start_time = time.time()

    rewards = defaultdict(list)
    player_types = defaultdict(list)
    allocations = defaultdict(list)
    payments = defaultdict(list)
    episode_lengths = []

    for sample_index in range(num_samples):
      if sample_index % report_freq == 0 and sample_index > 0:
        logging.info(f"----Episode {sample_index} ---")
        for player in range(num_players):
          avg_rewards = pd.Series(rewards[player]).mean()
          logging.info(f"Reward player {player}: {avg_rewards}")

      time_step = env.reset()
      episode_length = 0

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
        
        episode_length += 1
        player_id = time_step.observations["current_player"]
        agent = agents[player_id]
        agent_output = agent.step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      for i, agent in enumerate(agents):
        agent.step(time_step, is_evaluation=True)
        episode_rewards[i] += time_step.rewards[i] 
        rewards[i].append(episode_rewards[i])

        # Let's get allocation and pricing information since we're in the last time step
        infostate = time_step.observations['info_state'][i]
        payment, allocation = payment_and_allocation(num_players, num_actions, num_products, infostate)
        payments[i].append(payment)
        allocations[i].append(allocation)
        episode_lengths.append(episode_length)
    
    for player in range(num_players):
      logging.info(f"Rewards for {player}")
      logging.info(pd.Series(rewards[player]).describe())
      logging.info(f"-------------------")

    eval_time = time.time() - alg_start_time
    logging.info(f'Walltime: {pretty_time(eval_time)}')

    checkpoint = {
      'walltime': eval_time,
      'rewards': rewards, # For now, store all the rewards. But maybe we only need some summary stats. Or perhaps a counter is more compressed since few unique values in practice?
      'types': player_types,
      'allocations': allocations,
      'payments': payments,
      'auction_lengths': list((pd.Series(episode_lengths) / num_players))
    }
    return checkpoint

    
if __name__ == "__main__":
    app.run(main)
