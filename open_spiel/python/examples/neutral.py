from __future__ import absolute_import, division, print_function

import os
import shutil
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pyspiel
import torch
from absl import logging
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.agent_policy import NFSPPolicies
from open_spiel.python.examples.ubc_model_args import lookup_model_and_args
from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.pytorch import ubc_nfsp


@dataclass
class NFSPArgs:
    iterate_br: bool = False
    require_br: bool = False
    random_ic: bool = False

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
    sl_model, sl_model_args = lookup_model_and_args(config['sl_model'], state_size, num_actions, num_players, max_num_types(game_config), num_products)
    rl_model, rl_model_args = lookup_model_and_args(config['rl_model'], state_size, num_actions, num_players, max_num_types(game_config), num_products)

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
            cache_size=config['cache_size'],
            **dqn_kwargs
        )
        agents.append(agent)

    expl_policies_avg = NFSPPolicies(env, agents, False)
    return EnvAndModel(env=env, nfsp_policies=expl_policies_avg, agents=agents, game=game, game_config=game_config)
