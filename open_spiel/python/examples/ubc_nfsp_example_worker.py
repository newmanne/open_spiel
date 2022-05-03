from dataclasses import dataclass
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import ubc_nfsp
from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.ubc_model_args import lookup_model_and_args

import pandas as pd
import logging
import torch
import time
import os
import shutil
from typing import List
from torch.multiprocessing import Pool, current_process, Queue, Process, spawn, log_to_stderr, get_logger
from functools import partial
from open_spiel.python.examples.neutral import EnvAndModel, setup, NFSPArgs
from collections import defaultdict
import sys
import copy

class NFSPWorker(Process):

    def __init__(self, input_queue, output_queue, model_queue, game_name, nfsp_args):
        super(NFSPWorker, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_queue = model_queue
        self.game_name = game_name
        self.nfsp_args = nfsp_args
        
        # self.logger = get_logger()
        # self.logger.addHandler(logging.StreamHandler(sys.stderr))
        self.logger = log_to_stderr(logging.INFO)
        # self.logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s\n')
    # )


    def run(self):
        try:
            self.logger.info(f'Worker {current_process()} reporting for duty')

            # TODO:
            cfg = read_config('april26/mlp')
            self.logger.info(f'{current_process()} Done Reading config')
            cfg['num_training_episodes'] = 100
            cfg['device'] = 'cpu'

            # TODO: Should really load via the database...
            game = smart_load_clock_auction(self.game_name)
            game_config = load_game_config(self.game_name)
            self.env_and_model = setup(game, game_config, cfg)
            self.num_players = self.env_and_model.game.num_players()

            while True:
                start = time.time()
                data = self.input_queue.get()
                for d in data:
                    self.run_episode(d)
                self.logger.info(f"{current_process()} took {time.time() - start}")

                # for data in iter(self.input_queue.get, None):
                #     data = list(data)
                #     # Use 
                #     # self.logger.info(f'Worker {current_process()} is about to run episode {data}')
                #     for d in data:
                #         self.run_episode(d)

                # Feed data queue
                experience = defaultdict(dict)
                for agent in self.env_and_model.agents:
                    experience[agent.player_id]['resevoir'] = list(agent._reservoir_buffer._data)
                    experience[agent.player_id]['replay'] = list(agent._rl_agent._replay_buffer._data)
                    agent.clear_buffer()

                # self.logger.info(f'Worker {current_process()} is sharing their experience')
                self.output_queue.put(experience)

                # Wait for the new model parameters
                model_params = copy.deepcopy(self.model_queue.get())
                
                # Update all of the agents
                for k, v in model_params.items():
                    self.env_and_model.agents[k].restore(v)

            # TODO: Should probably have an ending condition
        except Exception as e:
            self.output_queue.put(e)

    def run_episode(self, ep):
        # self.logger.info(f"{current_process()} running episode {ep}")
        agents = self.env_and_model.agents
        env = self.env_and_model.env

        # Resample best response modes if needed
        if self.nfsp_args.require_br:
            while not any([agent._best_response_mode for agent in agents]):
                for agent in agents:
                    agent._sample_episode_policy()

        if self.nfsp_args.random_ic:
            current_eps = agents[0]._rl_agent._get_epsilon(False) 
            time_step = env.reset(current_eps)
        else:
            time_step = env.reset()

        episode_steps = 0 # TODO: Does this make any sense with random ICs?
        while not time_step.last():
            episode_steps += 1
            player_id = time_step.observations["current_player"]
            agent = agents[player_id]
            if self.nfsp_args.iterate_br: # Each player alternates between BR and Supervised network
                if ep % self.num_players == player_id:
                    with agent.temp_mode_as(True): # True=Best response mode
                        agent_output = agent.step(time_step)
                else:
                    agent_output = agent.step(time_step, is_evaluation=True) # Use supervised network and learn nothing
            else:
                agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        # episode_lengths.append(episode_steps)

        # Episode is over, step all agents with final info state.
        if self.nfsp_args.iterate_br:
            for player_id, agent in enumerate(agents):
                if ep % self.num_players == player_id:
                    # TODO: Temp mode as clearly wont work
                    with agent.temp_mode_as(True): 
                        agent.step(time_step)
                else:
                    agent.step(time_step, is_evaluation=True)
        else:
            for agent in agents:
                agent.step(time_step)

        # self.logger.info(f"Completed episode {ep}")

