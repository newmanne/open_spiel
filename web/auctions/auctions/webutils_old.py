
### OLD ###

def load_dqn_agent(best_response):
    # Takes a DB best response object and returns the DQN agent
    db_game = best_response.checkpoint.equilibrium_solver_run.game
    br_agent = make_dqn_agent(best_response.br_player, best_response.config, load_game(db_game), db_game.config)
    br_agent._q_network.load_state_dict(pickle.loads(best_response.model))
    if best_response.created < NORMALIZATION_DATE:
        br_agent._q_network.normalizer = torch.ones(10_000)
    return br_agent

def env_and_model_from_run(run):
    # Retrieve the game
    game_db_obj = run.game
    game = load_game(game_db_obj)
    game_config = game_db_obj.config

    # Get the NFSP config
    config = dict(run.config)

    # Create env_and_model
    env_and_model = setup(game, game_config, config)
    return env_and_model

def env_and_model_for_dry_run(game_db_obj, config):
    game = load_game(game_db_obj)
    game_config = game_db_obj.config
    config = dict(config)
    env_and_model = setup(game, game_config, config)
    return env_and_model


def db_checkpoint_loader(checkpoint):
    # Create an env_and_model based on an NFSP checkpoint in the database
    env_and_model = env_and_model_from_run(checkpoint.equilibrium_solver_run)

    # Restore the parameters
    nfsp_policies = env_and_model.nfsp_policies
    nfsp_policies.restore(pickle.loads(checkpoint.policy))

    if checkpoint.created < NORMALIZATION_DATE:
        for agent in env_and_model.agents:
            agent._rl_agent._q_network.normalizer = torch.ones(10_000)
            agent._avg_network.normalizer = torch.ones(10_000)

    return env_and_model

def load_game(game): # Takes in Django Game object
    return smart_load_sequential_game('clock_auction', dict(filename=game.name))
