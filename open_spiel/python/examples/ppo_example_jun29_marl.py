import argparse
import os
import random
import time
from distutils.util import strtobool
from datetime import datetime

# import gym
import pyspiel
from open_spiel.python.rl_environment import Environment
from open_spiel.python.examples.ubc_utils import UBCChanceEventSampler
import logging
from open_spiel.python.rl_agent import StepOutput
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def legal_actions_to_mask(legal_actions_list, num_actions):
    legal_actions_mask = torch.zeros((len(legal_actions_list), num_actions), dtype=torch.bool)
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask


def make_single_env(gym_id, seed, idx, capture_video, run_name, use_episodic_life_env=True):
    def thunk():
        game = pyspiel.load_game('atari', {
            'gym_id': gym_id, 
            'seed': seed, 
            'idx': idx, 
            'capture_video': capture_video, 
            'run_name': run_name, 
            'use_episodic_life_env': use_episodic_life_env
        })
        return Environment(game, chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=False, terminal_rewards=False)

    return thunk

class VectorEnv(object):
    """
    Greg change: wrapper to make OpenSpiel envs compatible with this code
    """
    def __init__(self, envs):
        self.envs = envs
        
    def __len__(self):
        return len(self.envs)

    def observation_spec(self):
        return self.envs[0].observation_spec()

    @property
    def num_players(self):
        return self.envs[0].num_players

    def step(self, step_outputs, reset_if_done=False):
        if not isinstance(step_outputs, list):
            step_outputs = [step_outputs]
        
        time_steps = [self.envs[i].step([step_outputs[i].action]) for i in range(len(self.envs))]
        reward = [step.rewards for step in time_steps]
        done = [step.last() for step in time_steps]
        raw_time_steps = time_steps # Copy these so you don't reset them

        if reset_if_done:
            time_steps = self.reset(envs_to_reset=done)

        return time_steps, reward, done, raw_time_steps

    def reset(self, envs_to_reset=None):
        if envs_to_reset is None:
            envs_to_reset = [True for _ in range(len(self.envs))]

        time_steps = [self.envs[i].reset() if envs_to_reset[i] else self.envs[i].get_time_step() for i in range(len(self.envs))]
        return time_steps


class EnvDecorator(object):

    _env: Environment = None

    def __init__(self, env: Environment) -> None:
        self._env = env
        self.env_attributes = [attribute for attribute in self._env.__dict__.keys()]
        self.env_methods = [m for m in dir(self._env) if not m.startswith('_') and m not in self.env_attributes]

    def __getattr__(self, func):
        if func in self.env_methods:
            def method(*args):
                return getattr(self._env, func)(*args)
            return method
        elif func in self.agent_attributes:
            return getattr(self._env, func)
        else:
            # For nesting decorators
            if isinstance(self._env, EnvDecorator):
                return self._env.__getattr__(func)
            raise AttributeError(func)

    def step(self, step_outputs):
        _ = self._env.step(step_outputs)
        return self.get_time_step()

    def reset(self):
        _ = self._env.reset()
        return self.get_time_step()

    @property
    def env(self) -> Environment:
        return self._env

class NormalizingEnvDecorator(EnvDecorator):

    def __init__(self, env: Environment, reward_normalizer: torch.tensor = None, info_state_normalizer: torch.tensor = None) -> None:
        super().__init__(env)
        self.reward_normalizer = reward_normalizer
        self.info_state_normalizer = info_state_normalizer
    
    def get_time_step(self):
        time_step = self._env.get_time_step()
        if self.reward_normalizer is not None:
            time_step.rewards[:] = (torch.tensor(time_step.rewards) / self.reward_normalizer).tolist() 

        if self.info_state_normalizer is not None:
            for p in len(time_step.observations['info_state']):
                time_step.observations['info_state'][p] = (torch.tensor(time_step.observations['info_state'][p]) / self.info_state_normalizer).tolist()

        return time_step


def setUpLogging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--game-name", type=str, default="atari",
        help="the id of the OpenSpiel game")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--eval-every", type=int, default=10,
        help="evaluate the policy every N updates")
    parser.add_argument("--eval-envs", type=int, default=4,
        help="the number of envs to evaluate the policy")
    parser.add_argument("--eval-episodes", type=int, default=100,
        help="the number of episodes to evaluate the policy")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def _eval_agent(env, agent, num_episodes, writer=None):
    """Evaluates `agent` for `num_episodes`."""
    if isinstance(env, VectorEnv):
        raise ValueError("Use the parallel function for VectorEnv.")
    
    rewards = []
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
            episode_reward += time_step.rewards[agent.player_id]
        rewards.append(episode_reward)

    if writer is not None:
        writer.add_histogram(
            f"evals/rewards", 
            np.array(rewards), 
            agent.total_steps_done
        )
        # logging.info(f'rewards: \n{pd.Series(rewards).value_counts()}')
    return sum(rewards) / num_episodes
    
def _eval_agent_parallel(env, agent, num_episodes, writer=None):
    """Evaluates `agent` for `num_episodes`."""
    if isinstance(env, VectorEnv):
        num_envs = len(env)
    else:
        num_envs = 1

    total_rewards = np.zeros(num_envs)
    episode_counter = num_envs
    time_step = env.reset()

    while episode_counter < num_episodes:
        agent_output = agent.step(time_step, is_evaluation=True)
        time_step, rewards, dones = env.step(agent_output, reset_if_done=True)
        total_rewards += np.array(rewards)[:, agent.player_id]
        episode_counter += sum(dones)

    finished = np.zeros(num_envs, dtype=bool)
    while not np.all(finished):
        agent_output = agent.step(time_step, is_evaluation=True)
        time_step, rewards, dones = env.step(agent_output, reset_if_done=True)
        total_rewards[~finished] += np.array(rewards)[~finished, agent.player_id]
        finished |= dones

    return total_rewards.sum() / episode_counter

class Agent(nn.Module):
    def __init__(self, num_actions, device):
        super(Agent, self).__init__()
        # TODO: How does it know input size?
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.num_actions = num_actions
        self.device = device

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            # All valid
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
        
        # TODO: This assumes the network is responsible for these funcitons....

        # Fill with invalids
        INVALID_ACTION_PENALTY = -1e6
        logits = torch.full((len(x), self.num_actions), INVALID_ACTION_PENALTY).to(self.device)
        hidden = self.network(x / 255.0)
        logits[legal_actions_mask] = self.actor(hidden)[legal_actions_mask]
        probs = Categorical(logits=logits)
            
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), probs.probs

# TODO: subclass rl_agent.AbstractAgent
class PPO(nn.Module):
    def __init__(
        self, 
        input_shape, 
        num_actions, 
        num_players,
        player_id=0,
        num_envs=1,
        steps_per_batch=128,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=2.5e-4, 
        num_annealing_updates=None, 
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        device='cpu', 
        writer=None # Tensorboard SummaryWriter
        ):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_players = num_players
        self.player_id = player_id
        self.device = device

        # Training settings
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.batch_size = self.num_envs * self.steps_per_batch
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.num_annealing_updates = num_annealing_updates

        # Loss function
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Logging
        self.writer = writer

        # Initialize networks
        # TODO: Clearly needs input size... Is it just hardcoded to atari?
        self.network = Agent(self.num_actions, device).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        
        # Initialize training buffers
        self.legal_actions_mask = torch.zeros((self.steps_per_batch, self.num_envs, self.num_actions), dtype=torch.bool).to(device)
        self.obs = torch.zeros((self.steps_per_batch, self.num_envs) + self.input_shape).to(device)
        self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)

        # Initialize counters 
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()

    def get_value(self, x):
        return self.network.get_value(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        return self.network.get_action_and_value(x, legal_actions_mask, action)

    def step(self, time_step, is_evaluation=False):
        if is_evaluation:
            singular_env = False
            if not isinstance(time_step, list):
                time_step = [time_step]
                singular_env = True

            with torch.no_grad():
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations['legal_actions'][self.player_id] for ts in time_step], self.num_actions
                ).to(self.device)
                obs = torch.Tensor(np.array([ts.observations['info_state'][self.player_id] for ts in time_step])).to(self.device)
                action, log_prob, entropy, value, probs = self.get_action_and_value(obs, legal_actions_mask=legal_actions_mask)

                # TODO: Probs
                if singular_env:
                    return StepOutput(action=action[0].item(), probs=probs[0])
                else:
                    return [StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)]
        else:
            with torch.no_grad():
                # act
                obs = torch.Tensor(np.array([ts.observations['info_state'][self.player_id] for ts in time_step])).to(self.device)
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations['legal_actions'][self.player_id] for ts in time_step], self.num_actions
                ).to(self.device)
                action, logprob, _, value, probs = self.get_action_and_value(obs, legal_actions_mask=legal_actions_mask)

                # store
                self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
                self.obs[self.cur_batch_idx] = obs
                self.actions[self.cur_batch_idx] = action
                self.logprobs[self.cur_batch_idx] = logprob
                self.values[self.cur_batch_idx] = value.flatten()

                agent_output = [StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)]
                return agent_output


    def post_step(self, reward, done):
        self.rewards[self.cur_batch_idx] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)

        self.total_steps_done += self.num_envs
        self.cur_batch_idx += 1

            
    def learn(self, time_step):
        next_obs = torch.Tensor(np.array([ts.observations['info_state'][self.player_id] for ts in time_step])).to(self.device)

        # Annealing the rate if instructed to do so.
        if self.num_annealing_updates is not None:
            frac = 1.0 - (self.updates_done) / self.num_annealing_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    nextvalues = next_value if t == self.steps_per_batch - 1 else self.values[t + 1]
                    nextnonterminal = 1.0 - self.dones[t]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = next_value if t == self.steps_per_batch - 1 else returns[t + 1]
                    nextnonterminal = 1.0 - self.dones[t]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.values

        # flatten the batch
        b_legal_actions = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(b_obs[mb_inds], legal_actions_mask=b_legal_actions[mb_inds], action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.total_steps_done)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps_done)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps_done)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps_done)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps_done)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps_done)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps_done)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.total_steps_done)
        self.writer.add_scalar("charts/SPS", int(self.total_steps_done / (time.time() - self.start_time)), self.total_steps_done)

        # Update counters 
        self.updates_done += 1
        self.cur_batch_idx = 0

def main():
    setUpLogging()
    args = parse_args()
    current_day = datetime.now().strftime('%d')
    current_month_text = datetime.now().strftime('%h')
    run_name = f"{args.game_name}__{args.gym_id}__{args.exp_name}__{args.seed}__{current_month_text}__{current_day}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"USING DEVICE: {device}")

    # env setup
    # (Greg change)
    envs = VectorEnv(
        [make_single_env(args.gym_id, args.seed + i, i, False, run_name)() for i in range(args.num_envs)]
    )

    if args.eval_envs == 0:
        pass
    elif args.eval_envs == 1:
        eval_env = make_single_env(args.gym_id, args.seed, 0, True, run_name, use_episodic_life_env=False)()
        eval_fn = _eval_agent
        logging.info("Using a single eval env")
    else:
        eval_env = VectorEnv(
            [make_single_env(args.gym_id, args.seed + i, i, True, run_name, use_episodic_life_env=False)() for i in range(args.eval_envs)]
        )
        eval_fn = _eval_agent_parallel
        logging.info("Using a parallel eval env")
    
    game = envs.envs[0]._game
    info_state_shape = tuple(np.array(envs.observation_spec()["info_state"]).flatten()) # TODO: does this return a tuple for flat infostates?
    num_updates = args.total_timesteps // args.batch_size
    agent = PPO(
        info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0, # TODO
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
    )

    time_step = envs.reset()
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
            agent_output = agent.step(time_step)
            time_step, reward, done, raw_time_step = envs.step(agent_output, reset_if_done=True)

            # Get around the fact that we use the life = episode idea in training but want to report true episode scores
            for ts in raw_time_step:
                info = ts.observations.get('info')
                if info and 'episode' in info:
                    real_reward = info['episode']['r']
                    writer.add_scalar('charts/player_0_training_returns', real_reward, agent.total_steps_done)

            agent.post_step(reward, done)

        # Learn
        agent.learn(time_step)

        # TODO: Maybe this concept just doesn't make sense in a single-player PPO env, and you would rather just report without an eval_env at all
        # Evaluate
        if args.eval_envs != 0 and update % args.eval_every == 0:
            logging.info("-" * 80)
            logging.info("Step %s", agent.total_steps_done)
            # logging.info("Loss: %s", loss.detach().cpu().numpy())
            avg_return = eval_fn(
                eval_env,
                agent, 
                args.eval_episodes,
                writer
            )
            logging.info("Avg return: %s", avg_return)
            writer.add_scalar('charts/player_0_avg_returns', avg_return, agent.total_steps_done)

    # envs.close()
    writer.close()
    logging.info("ALL DONE. GOODBYE. Have a pleasant day :)")


if __name__ == "__main__":
    main()