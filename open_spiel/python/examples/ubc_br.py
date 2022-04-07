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

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import ubc_dqn
from open_spiel.python.examples.ubc_utils import check_on_q_values, make_dqn_kwargs_from_config, fix_seeds, pretty_time, game_spec, make_normalizer_for_game, max_num_types
from open_spiel.python.examples.ubc_nfsp_example import lookup_model_and_args
from open_spiel.python.examples.ubc_decorators import CachingAgentDecorator
from open_spiel.python.algorithms.exploitability import best_response
import numpy as np
import pandas as pd
from absl import logging
import torch
import time
from open_spiel.python.pytorch import ubc_dqn

# TODO: Maybe want a convergence check here? Right now we just always run for a fixed number of episodes (which is indeed what e.g., the DREAM paper does)

def make_dqn_agent(player_id, config, game, game_config):
    num_players, num_actions, num_products = game_spec(game, game_config)
    state_size = rl_environment.Environment(game).observation_spec()["info_state"][0]

    rl_model, rl_model_args = lookup_model_and_args(config['rl_model'], state_size, num_actions, num_players, max_num_types(game_config), num_products)
    rl_model_args.update(config['rl_model_args'])
    normalizer = make_normalizer_for_game(game, game_config)
    rl_model_args.update({'normalizer': normalizer})
    dqn_kwargs = make_dqn_kwargs_from_config(config, game_config=game_config, player_id=player_id)

    return ubc_dqn.DQN(
        player_id,
        num_actions, 
        num_players,
        q_network_model=rl_model,
        q_network_args=rl_model_args,
        **dqn_kwargs
    )

def report(ep, agents, episode_lengths, br_player, game):
    logging.info(f"----Episode {ep} ---")
    loss = agents[br_player].loss
    logging.info(f"[P{br_player}] Loss: {loss}")

    logging.info(f"Episode length stats:\n{pd.Series(episode_lengths).describe()}")
    logging.info(check_on_q_values(agents[br_player], game))

def run_br(result_saver, report_freq, env_and_model, num_training_episodes, br_player, dry_run, seed, compute_exact_br, config):
    fix_seeds(seed) # This was probably done above (to deal with network initilization), but do it again for good measure
    alg_start_time = time.time()

    game, policy, env, trained_agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config

    # Prep agents
    agents = [CachingAgentDecorator(agent) for agent in trained_agents]
    agents[br_player] = make_dqn_agent(br_player, config, game, game_config)
    policy._policies[br_player] = agents[br_player]

    episode_lengths = []
    logging.info(f"Training for {num_training_episodes} episodes")

    # TRAINING PHASE
    for ep in range(num_training_episodes):
        for agent in agents:
            agent.set_global_iteration(ep)

        if ep % report_freq == 0 and ep > 1:
            report(ep, agents, episode_lengths, br_player, game)
            if compute_exact_br:
                logging.info("Computing exact BR")
                br = best_response(game, policy, br_player)
                gap = br['best_response_value'] - br['on_policy_value']
                logging.info(f"Gap between BR and current strategy: {gap}")

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
    # TODO: One might imagine saving out multiple checkpoints at multiple episodes to e.g. check if we training for the right number of iterations
    walltime_train = time.time() - alg_start_time
    checkpoint = {
      'br_player': br_player,
      'walltime': walltime_train,
      'agent': agents[br_player]._q_network.state_dict(),
      'config': config,
      'episode': ep
    }
    logging.info(f'Walltime: {pretty_time(walltime_train)}')

    if not dry_run:
        result_saver.save(checkpoint)

    logging.info('All done. Goodbye!')