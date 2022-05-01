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

from open_spiel.python.examples.ubc_nfsp_example_worker import NFSPWorker
from open_spiel.python.pytorch import ubc_nfsp
from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.ubc_model_args import lookup_model_and_args
import numpy as np
import pandas as pd
import logging
import torch
import time
import os
import shutil
from absl import logging
import torch.multiprocessing as mp
from open_spiel.python.examples.neutral import EnvAndModel, NFSPArgs, setup
from logging.handlers import QueueHandler, QueueListener
import gc
import copy

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



def run_nfsp(env_and_model, num_training_episodes, iterate_br, require_br, result_saver, seed, compute_nash_conv, dispatcher, eval_every, eval_every_early, eval_exactly, eval_zero, report_freq, dispatch_br, agent_selector, random_ic=False, parallelism=1):
    # This may have already been done, but do it again. Required to do it outside to ensure that networks get initilized the same way, which usually happens elsewhere
    fix_seeds(seed)
    game, policy, env, agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    num_players, num_actions, num_products = game_spec(game, game_config)

    nfsp_args = NFSPArgs(random_ic=random_ic, require_br=require_br, iterate_br=iterate_br)

    ### NFSP ALGORITHM
    nash_conv_history = []
    episode_lengths = []

    alg_start_time = time.time()

    N_PROCS = 1
    EPOCH_LENGTH = agents[0]._learn_every
    CHUNK_SIZE = 50 # THIS PROBABLY SHOULD BE A MUTLIPLE OF LEARN EVERY

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    model_queue = mp.Queue()

    try:

        for _ in range(N_PROCS):
            NFSPWorker(input_queue, output_queue, model_queue, game.get_parameters()['game']['filename'], nfsp_args).start()

        n_episodes = EPOCH_LENGTH // N_PROCS
        logging.info(f'Epochs will be {EPOCH_LENGTH} episodes. There will be {N_PROCS} processes each running {n_episodes} episodes per epoch')

        for epoch in range(1, (num_training_episodes // EPOCH_LENGTH) + 1):
            logging.info(f'Epoch {epoch}')

            for chunk_num in range(EPOCH_LENGTH // CHUNK_SIZE):
                chunk = range(chunk_num * CHUNK_SIZE, (chunk_num + 1) * CHUNK_SIZE)
                input_queue.put(chunk)
            
            # Put sentinels
            for _ in range(N_PROCS):
                input_queue.put(None)

            logging.info(f"{input_queue.qsize()} {output_queue.qsize()} {model_queue.qsize()}")

            # Retrieve data        
            data = []
            for _ in range(N_PROCS):
                result = copy.deepcopy(output_queue.get()) # Deepcoyp because otherwise we get OS:Error too many files and have no idea why 
                if isinstance(result, Exception):
                    raise result
                else:
                    data.append(result)

            # Add agent data
            for d in data:
                for pid in d:
                    agent = env_and_model.agents[pid]
                    for replay in d[pid]['replay']:
                        agent._rl_agent._replay_buffer.add(replay)
                    for resevoir in d[pid]['resevoir']:
                        agent._reservoir_buffer.add(resevoir)

            logging.info("Got data")
            model = dict()
            for agent in agents:
                agent._learn()
                agent._rl_agent.learn()
                model[agent.player_id] = agent.save()

            # epoch += 1

            # Send back model params
            for _ in range(N_PROCS):
                model_queue.put(model)

            logging.info(f'EPOCH {epoch} complete')

        # if self._iteration % self._update_target_network_every == 0 and self._last_network_copy < self._iteration:
        #   # logging.info(f"Copying target Q network for player {self.player_id} after {self._iteration} iterations")
        #   # state_dict method returns a dictionary containing a whole state of the module.
        #   self._target_q_network.load_state_dict(self._q_network.state_dict())
        #   self._last_network_copy = self._iteration

            ep = epoch * EPOCH_LENGTH
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
    finally:
        for p in mp.active_children():
            p.terminate()
        

    logging.info(f"Walltime: {pretty_time(time.time() - alg_start_time)}")
    logging.info('All done. Goodbye!')