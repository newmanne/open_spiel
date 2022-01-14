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

from dataclasses import dataclass
from open_spiel.python import rl_environment, policy
from open_spiel.python.pytorch import ubc_nfsp, ubc_dqn, ubc_rnn
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, clock_auction_bounds, check_on_q_values
from open_spiel.python.examples.ubc_nfsp_example import policy_from_checkpoint, lookup_model_and_args
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator
import pyspiel
import numpy as np
import pandas as pd
import copy
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
from typing import List
from open_spiel.python.pytorch import ubc_dqn
from pathlib import Path
import open_spiel.python.examples.ubc_dispatch as dispatch
from distutils import util

BR_DIR = 'best_responses'

# TODO: Maybe want a convergence check here? Right now we just always run for a fixed number of episodes (which is indeed what e.g., the DREAM paper does)

def checkpoint_sub_agent_i(experiment_dir, checkpoint_name, br_name):
  # Returns a policy with agent i subbed in with the best responder
  env_and_model = policy_from_checkpoint(experiment_dir, checkpoint_suffix=checkpoint_name)
  game, policy, env, trained_agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config

  with open(f'{experiment_dir}/{BR_DIR}/{br_name}.pkl', 'rb') as f:
    br_checkpoint = pickle.load(f)
    br_agent_id = br_checkpoint['br_player']
    br_config = br_checkpoint['config']
    br_agent = make_dqn_agent(br_agent_id, br_config, env, game, game_config)
    br_agent._q_network.load_state_dict(br_checkpoint['agent'])

  policy._policies[br_agent_id] = br_agent
  trained_agents[br_agent_id] = br_agent
  return policy


def make_dqn_agent(player_id, config, env, game, game_config):
  num_actions = env.action_spec()["num_actions"]
  num_players = game.num_players()
  num_products = len(game_config['activity'])
  state_size = env.observation_spec()["info_state"][0]

  rl_model, rl_model_args = lookup_model_and_args(config['rl_model'], state_size, num_actions, num_players, num_products)

  dqn_kwargs = {
    "replay_buffer_capacity": config['replay_buffer_capacity'],
    "epsilon_decay_duration": config['num_training_episodes'],
    "epsilon_start": config['epsilon_start'],
    "epsilon_end": config['epsilon_end'],
    "batch_size": config['batch_size'],
    "learning_rate": config['rl_learning_rate'],
    "learn_every": config['learn_every'],
    "min_buffer_size_to_learn": config['min_buffer_size_to_learn'],
    "optimizer_str": config['optimizer_str'],
    "update_target_network_every": config['update_target_network_every'],
    "loss_str": config['loss_str'],
  }

  dqn_kwargs['lower_bound_utility'], dqn_kwargs['upper_bound_utility'] = clock_auction_bounds(game_config, player_id)

  return ubc_dqn.DQN(
        player_id,
        num_actions, 
        num_players,
        q_network_model=rl_model,
        q_network_args=rl_model_args,
        **dqn_kwargs
  )

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_latest')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--br_player', type=int, default=0)
    parser.add_argument('--report_freq', type=int, default=50_000)
    parser.add_argument('--num_training_episodes', type=int, required=False)
    parser.add_argument('--dispatch_rewards', type=util.strtobool, default=0)
    parser.add_argument('--eval_overrides', type=str, default='')
    parser.add_argument('--output_name', type=str, default=None)

    args = parser.parse_args(argv[1:])  # Let argparse parse the rest of flags.

    experiment_dir = args.experiment_dir
    checkpoint_name = args.checkpoint
    br_player = args.br_player
    report_freq = args.report_freq
    dispatch_rewards = args.dispatch_rewards
    eval_overrides = args.eval_overrides
    output_name = args.output_name


    checkpoint_dir = os.path.join(experiment_dir, BR_DIR)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    logging.get_absl_handler().use_absl_log_file(f'br_{br_player}_{checkpoint_name}', checkpoint_dir) 
    logging.set_verbosity(logging.INFO)

    alg_start_time = time.time()

    env_and_model = policy_from_checkpoint(experiment_dir, checkpoint_suffix=checkpoint_name)
    game, policy, env, trained_agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    config_path = args.config if args.config is not None else f'{experiment_dir}/config.yml'

    with open(config_path, 'rb') as fh:
      config = yaml.load(fh, Loader=yaml.FullLoader)
    if args.num_training_episodes:
      config['num_training_episodes'] = args.num_training_episodes
    num_training_episodes = config['num_training_episodes']


    agents = []
    for i in range(game.num_players()):
      if i == br_player:
        agents.append(make_dqn_agent(i, config, env, game, game_config))
      else:
        agent = trained_agents[i]
        agent = CachingAgentDecorator(agent)
        agents.append(agent)

    episode_lengths = []
    logging.info(f"Training for {num_training_episodes} episodes")
    # TRAINING PHASE
    for i in range(num_training_episodes):
      if i % report_freq == 0:
        logging.info(f"----Episode {i} ---")
        loss = agents[br_player].loss
        logging.info(f"[P{br_player}] Loss: {loss}")
        logging.info(f"Episode length stats:\n{pd.Series(episode_lengths).describe()}")
        logging.info(check_on_q_values(agents[br_player], game))

      time_step = env.reset()
      episode_length = 0
      while not time_step.last():
        episode_length += 1
        player_id = time_step.observations["current_player"]
        agent = agents[player_id]
        agent_output = agent.step(time_step, is_evaluation=player_id != br_player)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      episode_lengths.append(episode_length)
      # Episode is over, step all agents with final info state.
      for player_id, agent in enumerate(agents):
        agent.step(time_step, is_evaluation=player_id != br_player)

    ### Save the best responding agent
    # TODO: We may want to save some indicators here to see if it is doing a good job of traininig?
    walltime_train = time.time() - alg_start_time
    br_name = f'{checkpoint_name}_br_{br_player}'
    checkpoint = {
      'br_player': br_player,
      'walltime': walltime_train,
      'agent': agents[br_player]._q_network.state_dict(),
      'config': config,
      'br_name': br_name
    }

    if output_name is None:
      output_name = br_name

    checkpoint_path = os.path.join(checkpoint_dir, f'{output_name}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    if dispatch_rewards:
      dispatch.dispatch_eval(experiment_dir, checkpoint=checkpoint_name, br_name=br_name, overrides=eval_overrides)

    logging.info('All done. Goodbye!')

    
if __name__ == "__main__":
    app.run(main)