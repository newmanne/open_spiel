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

from open_spiel.python import rl_environment, policy
from open_spiel.python.pytorch import ubc_nfsp
from open_spiel.python.algorithms.exploitability import nash_conv
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

class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = list(range(len(nfsp_policies)))
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None] * len(player_ids), "legal_actions": [None] * len(player_ids)}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='parameters.json')
    parser.add_argument('--network_config_file', type=str, default='network.yml')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--game_name', type=str, default='clock_auction')
    parser.add_argument('--turn_based', type=bool, default=True)

    # Optional Overrides
    parser.add_argument('--num_training_episodes', type=int, default=None)
    parser.add_argument('--replay_buffer_capacity', type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
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

    logging.info(f"Setting numpy and torch seed to {config['seed']}")
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    logging.info(f'Network params: {config}')

    start_time = time.time()
    logging.info("Loading %s", args.game_name)

    # LOAD GAME
    load_function = pyspiel.load_game if not args.turn_based else pyspiel.load_game_as_turn_based
    params = dict()
    if args.game_name == 'clock_auction':
        params['filename'] = args.filename

    game = load_function(args.game_name, params)

    logging.info("Game loaded")

    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    num_players = game.num_players()
    logging.info(f"Game has a state size of {state_size}, {num_actions} distinct actions, and {num_players} players")

    dqn_kwargs = {
      "replay_buffer_capacity": config['replay_buffer_capacity'],
      "epsilon_decay_duration": config['num_training_episodes'],
      "epsilon_start": config['epsilon_start'],
      "epsilon_end": config['epsilon_end'],
    }

    agents = []
    for player_id in range(game.num_players()):
        agent = ubc_nfsp.NFSP(
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=config['hidden_layers_sizes'],
            reservoir_buffer_capacity=config['reservoir_buffer_capacity'],
            anticipatory_param=config['anticipatory_param'],
            sl_learning_rate=config['sl_learning_rate'],
            rl_learning_rate=config['rl_learning_rate'],
            batch_size=config['batch_size'],
            min_buffer_size_to_learn=config['min_buffer_size_to_learn'],
            learn_every=config['learn_every'],
            optimizer_str=config['optimizer_str'],
            **dqn_kwargs
        )
        agents.append(agent)

    expl_policies_avg = NFSPPolicies(env, agents, ubc_nfsp.MODE.MODE_AVERAGE_POLICY)

    for ep in range(config['num_training_episodes']):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

        if (ep + 1) % config['eval_every'] == 0:
            losses = [agent.loss for agent in agents]
            logging.info("Losses: %s", losses)
            # pyspiel_policy = policy.python_policy_to_pyspiel_policy(expl_policies_avg)
            n_conv = nash_conv(game, expl_policies_avg, use_cpp_br=True)
            logging.info("[%s] NashConv AVG %s", ep + 1, n_conv)
            logging.info("_____________________________________________")

            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(expl_policies_avg, f)



if __name__ == "__main__":
    app.run(main)
