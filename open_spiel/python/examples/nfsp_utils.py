def read_nfsp_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    config['update_target_network_every'] = config.get('update_target_network_every', 10_000)
    config['loss_str'] = config.get('loss_str', 'mse')
    config['sl_loss_str'] = config.get('sl_loss_str', 'cross_entropy')
    config['double_dqn'] = config.get('double_dqn', True)
    config['device'] = config.get('device', default_device())
    config['anticipatory_param'] = config.get('anticipatory_param', 0.1)
    config['sl_learning_rate'] = config.get('sl_learning_rate', 0.01)
    config['rl_learning_rate'] = config.get('rl_learning_rate', 0.01)
    config['batch_size'] = config.get('batch_size', 256)
    config['learn_every'] = config.get('learn_every', 64)
    config['optimizer_str'] = config.get('optimizer_str', 'sgd')
    config['add_explore_transitions'] = config.get('add_explore_transitions', False)
    config['cache_size'] = config.get('cache_size', None)

    return config

def make_dqn_kwargs_from_config(config, game_config=None, player_id=None, include_nfsp=True):
    dqn_kwargs = {
      "replay_buffer_capacity": config['replay_buffer_capacity'],
      "epsilon_decay_duration": config['num_training_episodes'],
      "epsilon_start": config['epsilon_start'],
      "epsilon_end": config['epsilon_end'],
      "update_target_network_every": config['update_target_network_every'],
      "loss_str": config['loss_str'],
      "double_dqn": config['double_dqn'],
      "batch_size": config['batch_size'],
      "learning_rate": config['rl_learning_rate'],
      "learn_every": config['learn_every'],
      "min_buffer_size_to_learn": config['min_buffer_size_to_learn'],
      "optimizer_str": config['optimizer_str'],
      "device": config['device'],
      "cache_size": config['cache_size'],
    }
    if not include_nfsp:
        del dqn_kwargs['batch_size']
        del dqn_kwargs['min_buffer_size_to_learn']
        del dqn_kwargs['learn_every']
        del dqn_kwargs['optimizer_str']
        del dqn_kwargs['device']
        del dqn_kwargs['cache_size']

    if game_config is not None and player_id is not None:
        dqn_kwargs['lower_bound_utility'], dqn_kwargs['upper_bound_utility'] = clock_auction_bounds(game_config, player_id)
        dqn_kwargs['game_config'] = game_config
    return dqn_kwargs

def add_optional_overrides(parser):
    # Optional Overrides
    parser.add_argument('--replay_buffer_capacity', type=int, default=None)
    parser.add_argument('--reservoir_buffer_capacity', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--rl_learning_rate', type=float, default=None)
    parser.add_argument('--sl_learning_rate', type=float, default=None)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=None)
    parser.add_argument('--learn_every', type=int, default=None)
    parser.add_argument('--optimizer_str', type=str, default=None)
    parser.add_argument('--epsilon_start', type=float, default=None)
    parser.add_argument('--epsilon_end', type=float, default=None)

def check_on_q_values(agent, game, state=None, infostate_tensor=None, legal_actions=None, time_step=None, return_raw_q_values=False):

    if time_step is not None:
        legal_actions = time_step.observations["legal_actions"][agent.player_id]
        it = time_step.observations["info_state"][agent.player_id]
    elif infostate_tensor is not None:
        legal_actions = legal_actions
        it = infostate_tensor
    else:
        # Extract from state
        if state is None:
            # TODO: assuming player_id on agent is 0 here, could be smarter
            state = get_first_actionable_state(game, player_id=agent.player_id)
        legal_actions = state.legal_actions()
        it = state.information_state_tensor()

    q_values = agent.q_values_for_infostate(it)
    if return_raw_q_values:
        return q_values

    legal_q_values = q_values[legal_actions]
    action_dict = get_actions(game)
    return {s: q for s, q in zip(action_dict.values(), legal_q_values)}  
