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
from open_spiel.python import rl_environment, policy
from open_spiel.python.pytorch import ubc_nfsp, ubc_dqn, ubc_rnn, ubc_transformer
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, clock_auction_bounds
from open_spiel.python.algorithms.exploitability import nash_conv
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
import open_spiel.python.examples.ubc_dispatch as dispatch
from distutils import util
from copy import deepcopy

CHECKPOINT_FOLDER = 'solving_checkpoints' # Don't use "checkpoints" because jupyter bug

class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, best_response_mode):
        game = env.game
        player_ids = list(range(len(nfsp_policies)))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._best_response_mode = best_response_mode
        self._obs = {"info_state": [None] * len(player_ids), "legal_actions": [None] * len(player_ids)}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(observations=self._obs, rewards=None, discounts=None, step_type=None)

        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict

    def save(self):
        output = dict()
        for player, policy in enumerate(self._policies):
            output[player] = policy.save()
        return output

    def restore(self, restore_dict):
        for player, policy in enumerate(self._policies):
            policy.restore(restore_dict[player])

def lookup_model_and_args(model_name, state_size, num_actions, num_players, num_products):
    """
    lookup table from (model name) to (function, default args)
    TODO: cleaner way to do this?
    """

    if model_name == 'mlp': 
        model_class = ubc_dqn.MLP
        default_model_args = {
            'input_size': state_size,
            'hidden_sizes': [128],
            'output_size': num_actions,
        }
    elif model_name == 'recurrent':
        model_class = ubc_rnn.AuctionRNN
        default_model_args = {
            'num_players': num_players,
            'num_products': num_products, 
            'input_size': state_size, 
            'hidden_size': 128,
            'output_size': num_actions,
            'nonlinearity': 'tanh',
        }
    elif model_name == 'transformer': 
        model_class = ubc_transformer.AuctionTransformer
        default_model_args = {
            'num_players': num_players,
            'num_products': num_products,
            'input_size': state_size, 
            'output_size': num_actions, 
            'd_model': 32,
            'nhead': 1, 
            'feedforward_dim': 128, 
            'dropout': 0.1, 
            'num_layers': 1,
        }
    else: 
        raise ValueError(f'Unrecognized model {model_name}')
    
    return model_class, default_model_args


def policy_from_checkpoint(experiment_dir, checkpoint_suffix='checkpoint_latest'):
    with open(f'{experiment_dir}/config.yml', 'rb') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    env_and_model = setup(experiment_dir, config)

    nfsp_policies = env_and_model.nfsp_policies

    with open(f'{experiment_dir}/{CHECKPOINT_FOLDER}/{checkpoint_suffix}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    nfsp_policies.restore(checkpoint['policy'])
    return env_and_model

@dataclass
class EnvAndModel:
    env: rl_environment.Environment
    nfsp_policies: NFSPPolicies
    agents: List[ubc_nfsp.NFSP]
    game: pyspiel.Game
    game_config: dict

def setup(experiment_dir, config):
    if experiment_dir.endswith('/'):
        experiment_dir = experiment_dir[:-1]

    # Load game config
    game_config_path = f'{experiment_dir}/game.json'
    with open(game_config_path, 'r') as f:
        game_config = json.load(f)

    # Load game
    game = smart_load_sequential_game('clock_auction', dict(filename=game_config_path))
    logging.info("Game loaded")

    env = rl_environment.Environment(game)
    if not env.is_turn_based:
      raise ValueError("Expected turn based env")
    
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = game.num_players()
    num_products = len(game_config['activity'])
    logging.info(f"Game has a state size of {state_size}, {num_actions} distinct actions, and {num_players} players")
    logging.info(f"Game has {num_products} products")

    dqn_kwargs = {
      "replay_buffer_capacity": config['replay_buffer_capacity'],
      "epsilon_decay_duration": config['num_training_episodes'],
      "epsilon_start": config['epsilon_start'],
      "epsilon_end": config['epsilon_end'],
      "update_target_network_every": config.get('update_target_network_every', 1_000),
      "loss_str": config.get('loss_str', 'mse'),
      "weight_iters": config.get('weight_iters_dqn', False),
    }

    # Get models and default args
    sl_model, sl_model_args = lookup_model_and_args(config['sl_model'], state_size, num_actions, num_players, num_products)
    rl_model, rl_model_args = lookup_model_and_args(config['rl_model'], state_size, num_actions, num_players, num_products)

    # Override with any user-supplied args
    sl_model_args.update(config['sl_model_args'])
    rl_model_args.update(config['rl_model_args'])

    agents = []
    for player_id in range(game.num_players()):
        dqn_kwargs_copy = copy.deepcopy(dqn_kwargs)
        dqn_kwargs_copy['lower_bound_utility'], dqn_kwargs_copy['upper_bound_utility'] = clock_auction_bounds(game_config, player_id)
        
        agent = ubc_nfsp.NFSP(
            player_id,
            num_actions=num_actions,
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
            add_explore_transitions=config.get('add_explore_transitions', True),
            **dqn_kwargs_copy
        )
        agents.append(agent)

    expl_policies_avg = NFSPPolicies(env, agents, False)
    return EnvAndModel(env=env, nfsp_policies=expl_policies_avg, agents=agents, game=game, game_config=game_config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='parameters.json')
    parser.add_argument('--network_config_file', type=str, default='network.yml')
    parser.add_argument('--output_dir', type=str, default='output') # Note: DONT NAME THIS "checkpoints" because of a jupyter notebook
    parser.add_argument('--game_name', type=str, default='clock_auction')
    parser.add_argument('--job_name', type=str, default='auction')
    parser.add_argument('--warn_on_overwrite', type=bool, default=False)
    parser.add_argument('--compute_nash_conv', type=bool, default=False)
    parser.add_argument('--eval_every', type=int, default=300_000)
    parser.add_argument('--dispatch_br', type=util.strtobool, default=0)
    parser.add_argument('--br_overrides', type=str, default='')
    parser.add_argument('--eval_overrides', type=str, default='')
    parser.add_argument('--report_freq', type=int, default=50_000)
    parser.add_argument('--iterate_br', type=util.strtobool, default=0)

    # Optional Overrides
    parser.add_argument('--num_training_episodes', type=int, default=None)
    parser.add_argument('--replay_buffer_capacity', type=int, default=None)
    parser.add_argument('--reservoir_buffer_capacity', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--rl_learning_rate', type=float, default=None)
    parser.add_argument('--sl_learning_rate', type=float, default=None)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=None)
    parser.add_argument('--learn_every', type=int, default=None)
    parser.add_argument('--optimizer_str', type=str, default=None)
    parser.add_argument('--epsilon_start', type=float, default=None)
    parser.add_argument('--epsilon_end', type=float, default=None)

    args = parser.parse_args(argv[1:])  # Let argparse parse the rest of flags.
    output_dir = args.output_dir
    dispatch_br = args.dispatch_br
    br_overrides = args.br_overrides
    eval_overrides = args.eval_overrides
    report_freq = args.report_freq
    iterate_br = args.iterate_br


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if args.warn_on_overwrite:
            raise ValueError("You are overwriting a folder!")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    os.makedirs(os.path.join(output_dir, CHECKPOINT_FOLDER))

    logging.get_absl_handler().use_absl_log_file('nfsp', output_dir) 
    logging.set_verbosity(logging.INFO)

    if iterate_br:
        logging.info("Iterating best responses")
    
    logging.info("Loading network parameters from %s", args.network_config_file)

    with open(args.network_config_file, 'rb') as fh:
        config = yaml.load(fh,Loader=yaml.FullLoader)

    # Override any top-level yaml args with command line arguments
    for arg in vars(args):
        if f'--{arg}' in sys.argv:
            name = arg
            value = getattr(args, arg)

            if name in config:
                config[name] = value

    # Save the final overridden config so there's no confusion later if you need to cross-reference
    with open(f'{output_dir}/config.yml', 'w') as outfile:
        yaml.dump(config, outfile)

    logging.info(f"Setting numpy and torch seed to {config['seed']}")
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    logging.info(f'Network params: {config}')

    start_time = time.time()

    # additionally, load game params to set up models
    game_config_path = os.path.join(os.environ['CLOCK_AUCTION_CONFIG_DIR'], args.filename)
    with open(game_config_path, 'r') as f:
        game_config = json.load(f)
    # Save the game config so there's no confusion later if you need to cross-reference
    with open(f'{output_dir}/game.json', 'w') as outfile:
        json.dump(game_config, outfile)

    logging.info("Loading %s", args.game_name)

    env_and_model = setup(output_dir, config)
    env = env_and_model.env
    agents = env_and_model.agents
    nfsp_policies = env_and_model.nfsp_policies
    game = env_and_model.game
    num_players = game.num_players()

    ### NFSP ALGORITHM
    compute_nash_conv = args.compute_nash_conv
    nash_conv_history = []
    min_nash_conv = None

    alg_start_time = time.time()
    for ep in range(1, config['num_training_episodes'] + 1):
        if ep % report_freq == 0:
            logging.info(f"----Episode {ep} ---")

        time_step = env.reset()
        while not time_step.last():
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

        if ep % config['eval_every'] == 0 or ep == config['num_training_episodes']:
            losses = [agent.loss for agent in agents]
            logging.info(f"[{ep}] Losses: {losses}")

            if compute_nash_conv:
                logging.info('Computing nash conv...')
                n_conv = nash_conv(game, nfsp_policies, use_cpp_br=True)
                logging.info("[%s] NashConv AVG %s", ep, n_conv)
                logging.info("_____________________________________________")
            else:
                n_conv = None

            nash_conv_history.append((ep, time.time() - alg_start_time, n_conv))

            checkpoint = {
                    'name': args.job_name,
                    'walltime': time.time() - alg_start_time,
                    'policy': nfsp_policies.save(),
                    'nash_conv_history': nash_conv_history,
                }

            checkpoint_path = os.path.join(output_dir, CHECKPOINT_FOLDER, 'checkpoint_latest.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            checkpoint_name = f'checkpoint_{ep}'
            shutil.copyfile(checkpoint_path, os.path.join(output_dir, CHECKPOINT_FOLDER, f'{checkpoint_name}.pkl'))
            
            ### DISPATCH BEST RESPONSE JOBS
            if dispatch_br:
                for player in range(game.num_players()):
                    dispatch.dispatch_br(output_dir, br_player=player, checkpoint=checkpoint_name, overrides=br_overrides + f' --eval_overrides "{eval_overrides}"')
                dispatch.dispatch_eval(output_dir, checkpoint=checkpoint_name, overrides=eval_overrides)

            if compute_nash_conv:
                best_checkpoint_path = os.path.join(output_dir, CHECKPOINT_FOLDER, 'checkpoint_best.pkl')
                if min_nash_conv is None or n_conv <= min_nash_conv:
                    min_nash_conv = n_conv
                    shutil.copyfile(checkpoint_path, best_checkpoint_path)

    logging.info('All done. Goodbye!')


if __name__ == "__main__":
    app.run(main)
