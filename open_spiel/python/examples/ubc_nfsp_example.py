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

"""Tests for open_spiel.python.algorithms.nfsp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import dataclass
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import ubc_nfsp
from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.ubc_model_args import lookup_model_and_args
import pyspiel
from open_spiel.python.examples.agent_policy import NFSPPolicies
import numpy as np
import pandas as pd
from absl import logging
import torch
import time
import os
import shutil
from typing import List

@dataclass
class EnvAndModel:
    env: rl_environment.Environment
    nfsp_policies: NFSPPolicies
    agents: List[ubc_nfsp.NFSP]
    game: pyspiel.Game
    game_config: dict

def setup(game, game_config, config):
    env = rl_environment.Environment(game, chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=True, terminal_rewards=True)
    if not env.is_turn_based:
      raise ValueError("Expected turn based env")
    
    state_size = env.observation_spec()["info_state"][0]
    num_players, num_actions, num_products = game_spec(game, game_config)
    logging.info(f"Game has a state size of {state_size}, {num_actions} distinct actions, and {num_players} players")
    logging.info(f"Game has {num_products} products")

    # Get models and default args
    sl_model, sl_model_args = lookup_model_and_args(config['sl_model'], state_size, num_actions, num_players, num_products)
    rl_model, rl_model_args = lookup_model_and_args(config['rl_model'], state_size, num_actions, num_players, num_products)

    # Override with any user-supplied args
    sl_model_args.update(config['sl_model_args'])
    rl_model_args.update(config['rl_model_args'])

    normalizer = make_normalizer_for_game(game, game_config)
    sl_model_args.update({'normalizer': normalizer})
    rl_model_args.update({'normalizer': normalizer})

    agents = []
    for player_id in range(num_players):
        dqn_kwargs = make_dqn_kwargs_from_config(config, game_config=game_config, player_id=player_id, include_nfsp=False)

        agent = ubc_nfsp.NFSP(
            player_id,
            num_actions=num_actions,
            num_players=num_players,
            sl_model=sl_model,
            sl_model_args=sl_model_args,
            rl_model=rl_model,
            rl_model_args=rl_model_args,
            reservoir_buffer_capacity=config['reservoir_buffer_capacity'],
            anticipatory_param=config['anticipatory_param'],
            sl_learning_rate=config['sl_learning_rate'],
            rl_learning_rate=config['rl_learning_rate'],
            batch_size=config['batch_size'],
            min_buffer_size_to_learn=config['min_buffer_size_to_learn'],
            learn_every=config['learn_every'],
            optimizer_str=config['optimizer_str'],
            add_explore_transitions=config['add_explore_transitions'],
            device=config['device'],
            sl_loss_str=config['sl_loss_str'],
            **dqn_kwargs
        )
        agents.append(agent)

    expl_policies_avg = NFSPPolicies(env, agents, False)
    return EnvAndModel(env=env, nfsp_policies=expl_policies_avg, agents=agents, game=game, game_config=game_config)


def setup_directory_structure(output_dir, warn_on_overwrite, database=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if warn_on_overwrite:
            raise ValueError("You are overwriting a folder!")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    os.makedirs(os.path.join(output_dir, BR_DIR))
    os.makedirs(os.path.join(output_dir, EVAL_DIR))
    if not database:
        os.makedirs(os.path.join(output_dir, CHECKPOINT_FOLDER))

def report_nfsp(ep, episode_lengths, num_players, agents, game, start_time):
    logging.info(f"----Episode {ep} ---")
    logging.info(f"Episode length stats:\n{pd.Series(episode_lengths).describe()}")
    for player_id in range(num_players):
        logging.info(f"PLAYER {player_id}")
        agent = agents[player_id]
        if isinstance(agent, ubc_nfsp.NFSP):
            logging.info(check_on_q_values(agent._rl_agent, game))
            logging.info(f"Train time {pretty_time(agent._rl_agent._train_time)}")
            logging.info(f"Total time {pretty_time(time.time() - start_time)}")
            logging.info(f"Training the DQN for player {player_id} is a {agent._rl_agent._train_time / (time.time() - start_time):.2f} fraction")
        logging.info(f'Loss (Supervised, RL): {agent.loss}')

def evaluate_nfsp(ep, compute_nash_conv, game, policy, alg_start_time, nash_conv_history):
    logging.info(f"EVALUATION AT ITERATION {ep}")
    if compute_nash_conv:
        logging.info('Computing nash conv...')
        n_conv = nash_conv(game, policy, use_cpp_br=True)
        logging.info("[%s] NashConv AVG %s", ep, n_conv)
        logging.info("_____________________________________________")
    else:
        n_conv = None

    nash_conv_history.append((ep, time.time() - alg_start_time, n_conv))

    checkpoint = {
            'walltime': time.time() - alg_start_time,
            'policy': policy.save(),
            'nash_conv_history': nash_conv_history,
            'episode': ep,
        }
    return checkpoint

def run_nfsp(env_and_model, num_training_episodes, iterate_br, result_saver, seed, compute_nash_conv, dispatcher, eval_every, eval_every_early, eval_exactly, eval_zero, report_freq, dispatch_br, agent_selector):
    # This may have already been done, but do it again. Required to do it outside to ensure that networks get initilized the same way, which usually happens elsewhere
    fix_seeds(seed)
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    ### NFSP ALGORITHM
    nash_conv_history = []
    episode_lengths = []

    alg_start_time = time.time()
    for ep in range(1, num_training_episodes + 1):

        # Start of new episode bookkeeping
        for agent in agents:
            agent.set_global_iteration(ep)
        if agent_selector is not None:
            agent_selector.new_episode(ep) 

        time_step = env.reset()
        episode_steps = 0

        while not time_step.last():
            episode_steps += 1
            player_id = time_step.observations["current_player"]
            agent = agents[player_id]
            if iterate_br: # Each player alternates between BR and Supervised network
                if ep % num_players == player_id:
                    with agent.temp_mode_as(True): # True=Best response mode
                        agent_output = agent.step(time_step)
                else:
                    agent_output = agent.step(time_step, is_evaluation=True) # Use supervised network and learn nothing
            else:
                agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        episode_lengths.append(episode_steps)

        # Episode is over, step all agents with final info state.
        if iterate_br:
            for player_id, agent in enumerate(agents):
                if ep % num_players == player_id:
                    with agent.temp_mode_as(True): 
                        agent.step(time_step)
                else:
                    agent.step(time_step, is_evaluation=True)
        else:
            for agent in agents:
                agent.step(time_step)

        eval_step = eval_every
        if eval_every_early is not None and ep < eval_every:
            eval_step = eval_every_early
        should_eval = ep % eval_step == 0 or ep == num_training_episodes or ep in eval_exactly or ep == 1 and eval_zero
        should_report = should_eval or (ep % report_freq == 0 and ep > 1)

        if should_report:
            report_nfsp(ep, episode_lengths, num_players, agents, game, alg_start_time)
        if should_eval:
            checkpoint = evaluate_nfsp(ep, compute_nash_conv, game, policy, alg_start_time, nash_conv_history)
            checkpoint_name = result_saver.save(checkpoint)
            if dispatch_br:
                dispatcher.dispatch(checkpoint_name)

    logging.info(f"Walltime: {pretty_time(time.time() - alg_start_time)}")
    logging.info('All done. Goodbye!')