# Note: code adapted (with permission) from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py

import argparse
import collections
import logging
import os
import random
import sys
import time
from datetime import datetime
from distutils.util import strtobool
import numpy as np
import pandas as pd
import pyspiel
import torch
from open_spiel.python.pytorch.ppo import PPO, PPOAtariAgent, PPOAgent
from open_spiel.python.rl_agent import StepOutput
from open_spiel.python.rl_environment import Environment, ChanceEventSampler
from open_spiel.python.vector_env import SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
import open_spiel.python.games
from open_spiel.python.examples.ppo_eval import eval_agents_parallel, eval_agent
from open_spiel.python.env_decorator import NormalizingEnvDecorator
from open_spiel.python.examples.ubc_utils import clock_auction_bounds, load_game_config
from open_spiel.python.examples.ubc_utils import UBCChanceEventSampler
from open_spiel.python.rl_agent_policy import JointRLAgentPolicy
from open_spiel.python.algorithms.exploitability import nash_conv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--game-name", type=str, default="python_clock_auction",
        help="the id of the OpenSpiel game")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--eval-every", type=int, default=10,
        help="evaluate the policy every N updates")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--eval-envs", type=int, default=4,
        help="the number of envs to evaluate the policy")
    parser.add_argument("--eval-episodes", type=int, default=100,
        help="the number of episodes to evaluate the policy")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def setUpLogging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def make_single_env(game_name, seed, normalize_rewards=True):
    if game_name == 'python_clock_auction':
        def gen_env():
            auction_game = 'small_game_2.json'
            game = pyspiel.load_game('python_clock_auction', dict(filename=auction_game))
            env = Environment(game, chance_event_sampler=UBCChanceEventSampler(seed=seed))
            if normalize_rewards:
                env = NormalizingEnvDecorator(env, reward_normalizer=torch.tensor(game.upper_bounds))
            return env
    else:
        def gen_env():
            game = pyspiel.load_game(game_name)
            return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))
    return gen_env


def main():
    setUpLogging()
    args = parse_args()

    current_day = datetime.now().strftime('%d')
    current_month_text = datetime.now().strftime('%h')
    run_name = f"{args.game_name}__{args.exp_name}__"
    run_name += f"{args.seed}__{current_month_text}__{current_day}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
 
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    envs = SyncVectorEnv(
        [make_single_env(args.game_name, args.seed + i)() for i in range(args.num_envs)]
    )
    agent_fn = PPOAgent
    if args.eval_envs == 0:
        pass
    elif args.eval_envs == 1:
        eval_env = make_single_env(args.game_name, args.seed)()
        eval_fn = eval_agent
        logging.info("Using a single eval env")
    else:
        eval_env = SyncVectorEnv(
            [make_single_env(args.game_name, args.seed + i, normalize_rewards=False)() for i in range(args.eval_envs)]
        )
        eval_fn = eval_agents_parallel
        logging.info("Using a parallel eval env")

    game = envs.envs[0]._game
    info_state_shape = tuple(np.array(envs.observation_spec()["info_state"]).flatten()) 
    num_updates = args.total_timesteps // args.batch_size

    agents = []
    for i in range(game.num_players()):
        agent = PPO(
            input_shape=info_state_shape,
            num_actions=game.num_distinct_actions(),
            num_players=game.num_players(),
            player_id=i,
            num_envs=args.num_envs,
            steps_per_batch=args.num_steps,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            num_annealing_updates=num_updates,
            gae=args.gae,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=args.norm_adv,
            clip_coef=args.clip_coef,
            clip_vloss=args.clip_vloss,
            entropy_coef=args.ent_coef,
            value_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            device=device,
            writer=writer,
            agent_fn=agent_fn,
        )
        agents.append(agent)

    N_REWARD_WINDOW = 50
    recent_rewards = [collections.deque(maxlen=N_REWARD_WINDOW) for i in range(game.num_players())]
    time_step = envs.reset()
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            for agent in agents: # ENFORCED ORDERING HERE - I'm delibarately not checking time steps
                agent_output = agent.step(time_step)
                time_step, reward, done, unreset_time_steps = envs.step(agent_output, reset_if_done=True)

            for player_id, agent in enumerate(agents):  
                agent.post_step([r[player_id] for r in reward], done)

            for ts in unreset_time_steps:
                if ts.last():
                    for player_id, agent in enumerate(agents):
                        real_reward = ts.rewards[player_id]
                        writer.add_scalar(f'charts/player_{player_id}_training_returns', real_reward, agent.total_steps_done)
                        recent_rewards[player_id].append(real_reward)

        for agent in agents:
            agent.learn(time_step)

        if update % args.eval_every == 0:
            # TODO: Do real evals!!! With approx nash conv and holding the other players fixed
            logging.info("-" * 80)
            logging.info("Step %s", agent.total_steps_done)
            avg_return = eval_fn(
                eval_env,
                agents, 
                args.eval_episodes,
                writer
            )
            for player_id, agent in enumerate(agents):
                agent_return = avg_return[player_id]
                logging.info(f"Avg return for Player {player_id}: {agent_return}")
                writer.add_scalar('charts/player_0_avg_returns', agent_return, agent.total_steps_done)

            
            compute_exact_nash_conv = True
            if compute_exact_nash_conv:
                logging.info('Computing nash conv...')
                agent_dict = {agent.player_id: agent for agent in agents}
                policy = JointRLAgentPolicy(game, agent_dict, False)
                n_conv = nash_conv(game, policy, use_cpp_br=True)
                logging.info(f"{n_conv}")

    writer.close()
    logging.info("All done. Have a pleasant day :)")


if __name__ == "__main__":
    main()
