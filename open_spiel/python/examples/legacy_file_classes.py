import os
import pickle
import shutil
import open_spiel.python.examples.legacy_ubc_dispatch as dispatch

class FileNFSPResultSaver:

    def __init__(self, output_dir, job_name):
        self.output_dir = output_dir
        self.job_name = job_name

    def save(self, result):
        result['name'] = self.job_name
        checkpoint_path = os.path.join(self.output_dir, CHECKPOINT_FOLDER, 'checkpoint_latest.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(result, f)

        checkpoint_name = f'checkpoint_{result["episode"]}'
        shutil.copyfile(checkpoint_path, os.path.join(self.output_dir, CHECKPOINT_FOLDER, f'{checkpoint_name}.pkl'))
        return checkpoint_name


class FileDispatcher:

    def __init__(self, num_players, output_dir, br_overrides, eval_overrides, br_portfolio_path):
        self.num_players = num_players
        self.output_dir = output_dir
        self.br_overrides = br_overrides
        self.eval_overrides = eval_overrides
        self.br_portfolio_path = br_portfolio_path

    def dispatch(self, checkpoint_name):
        for player in range(self.num_players):
            dispatch.dispatch_br(self.output_dir, br_player=player, checkpoint=checkpoint_name, overrides=self.br_overrides + f' --eval_overrides "{self.eval_overrides}"', br_portfolio_path=self.br_portfolio_path)
            dispatch.dispatch_eval(self.output_dir, checkpoint=checkpoint_name, straightforward_player=player, overrides=self.eval_overrides)
        dispatch.dispatch_eval(self.output_dir, checkpoint=checkpoint_name, overrides=eval_overrides)

class BRFileResultSaver:

    def __init__(self, checkpoint_dir, pickle_path, br_name):
        self.checkpoint_dir = checkpoint_dir
        self.pickle_path = pickle_path
        self.br_name = br_name

    def save(self, br_result):
        pickle_path = self.pickle_path
        if pickle_path is None:
            pickle_path = os.path.join(self.checkpoint_dir, f'{self.br_name}.pkl')

        logging.info(f'Pickling model to {pickle_path}')
        with open(pickle_path, 'wb') as f:
            pickle.dump(br_result, f)


def dqn_agent_from_checkpoint(experiment_dir, checkpoint_name, br_name):
    # Deprecated, use load_dqn_agent for database
    env_and_model = policy_from_checkpoint(experiment_dir, checkpoint_suffix=checkpoint_name)
    game, policy, env, trained_agents, game_config = env_and_model.game, env_and_model.nfsp_policies, env_and_model.env, env_and_model.agents, env_and_model.game_config
    with open(f'{experiment_dir}/{BR_DIR}/{br_name}.pkl', 'rb') as f:
        br_checkpoint = pickle.load(f)
        br_agent_id = br_checkpoint['br_player']
        br_config = br_checkpoint['config']
        br_agent = make_dqn_agent(br_agent_id, br_config, game, game_config)
        br_agent._q_network.load_state_dict(br_checkpoint['agent'])
    return br_agent

def policy_from_checkpoint(experiment_dir, checkpoint_suffix='checkpoint_latest'):
    with open(f'{experiment_dir}/config.yml', 'rb') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if experiment_dir.endswith('/'):
        experiment_dir = experiment_dir[:-1]

    # Load game config
    game_config_path = f'{experiment_dir}/game.json'
    with open(game_config_path, 'r') as f:
        game_config = json.load(f)

    # Load game
    game = smart_load_sequential_game('clock_auction', dict(filename=str(Path(game_config_path).resolve())))
    logging.info("Game loaded")

    env_and_model = setup(game, game_config, config)

    nfsp_policies = env_and_model.nfsp_policies

    with open(f'{experiment_dir}/{CHECKPOINT_FOLDER}/{checkpoint_suffix}.pkl', 'rb') as f:
        checkpoint = pickle.load(f)

    nfsp_policies.restore(checkpoint['policy'])
    return env_and_model