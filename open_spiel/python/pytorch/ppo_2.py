import argparse
import os
import random
import time
from distutils.util import strtobool

# import gym
import pyspiel
from open_spiel.python.rl_environment import Environment
from open_spiel.python.examples.ubc_utils import UBCChanceEventSampler, fix_seeds
import logging
from open_spiel.python.examples.single_agent_catch import _eval_agent
from open_spiel.python.rl_agent import StepOutput
import sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

class VectorEnv(object):
    """
    Greg change: wrapper to make OpenSpiel envs compatible with this code
    """
    def __init__(self, envs):
        self.envs = envs

    def observation_spec(self):
        return self.envs[0].observation_spec()
        

    def step(self, actions, reset_if_done=False):
        player_id = 0
        time_steps = [self.envs[i].step([actions[i]]) for i in range(len(self.envs))]
        reward = [step.rewards[player_id] for step in time_steps]
        done = [step.last() for step in time_steps]

        if reset_if_done:
            time_steps = self.reset(envs_to_reset=done)

        return time_steps, reward, done

    def reset(self, envs_to_reset=None):
        if envs_to_reset is None:
            envs_to_reset = [True for _ in range(len(self.envs))]

        time_steps = [self.envs[i].reset() if envs_to_reset[i] else self.envs[i].get_time_step() for i in range(len(self.envs))]
        return time_steps


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
    parser.add_argument("--game-name", type=str, default="catch",
        help="the name of the OpenSpiel game")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
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
    parser.add_argument("--clip-coef", type=float, default=0.2,
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

def _eval_agent(env, agent, num_episodes):
    """Evaluates `agent` for `num_episodes`."""
    rewards = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step = env.step(agent_output.action)
            episode_reward += time_step.rewards[0]
        rewards += episode_reward
    return rewards / num_episodes

class PPO(nn.Module):
    def __init__(
        self, 
        input_size, 
        num_actions, 
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
        writer=None
    ):
        super(PPO, self).__init__()

        # Networks
        # TODO: move models to a separate class?
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(device)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        ).to(device)
        self.input_size = input_size
        self.device = device

        # Training details
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.num_minibatches = num_minibatches
        self.batch_size = int(num_envs * steps_per_batch)
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.update_epochs = update_epochs
        self.initial_learning_rate = learning_rate
        self.num_annealing_updates = num_annealing_updates
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

        # PPO settings
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

        # Initial values
        self.current_step = 0
        self.updates_done = 0
        self.total_steps_done = 0

        # Training data
        self.obs      = torch.zeros((self.steps_per_batch, self.num_envs, self.input_size)).to(device)
        self.actions  = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards  = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones    = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values   = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)

        # Logging
        self.writer = writer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def step(self, time_steps, is_evaluation=False):
        if not isinstance(time_steps, list):
            time_steps = [time_steps]

        obs = torch.Tensor([ts.observations['info_state'][0] for ts in time_steps]).to(self.device)
        with torch.no_grad():
            if is_evaluation:
                # Evaluation: no need track data; just sample an action
                action, _, _, _ = self.get_action_and_value(obs)
                return StepOutput(action=[action.item()], probs=None)
            else:
                # Sample action
                action, logprob, _, value = self.get_action_and_value(obs)

                # Store 
                self.obs[self.current_step] = obs
                self.values[self.current_step] = value.flatten()
                self.actions[self.current_step] = action
                self.logprobs[self.current_step] = logprob

                # Note: don't update step count until post_step!
                return action

    def post_step(self, reward, done):
        self.rewards[self.current_step] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[self.current_step] = torch.tensor(done).to(self.device).view(-1)

        self.current_step += 1
        self.total_steps_done += self.num_envs

    def learn(self, next_time_steps):
        if not isinstance(next_time_steps, list):
            next_time_steps = [next_time_steps]

        # Anneal learning rate
        if self.num_annealing_updates is not None:
            frac = 1.0 - (self.updates_done) / self.num_annealing_updates
            lrnow = frac * self.initial_learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Bootstrap values for non-terminal states
        with torch.no_grad():
            next_obs = torch.Tensor([ts.observations['info_state'][0] for ts in next_time_steps]).to(self.device)
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    next_values = next_value if (t == self.steps_per_batch - 1) else self.values[t+1]
                    is_nonterminal = 1.0 - self.dones[t]
                    delta = self.rewards[t] + self.gamma * next_values * is_nonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * is_nonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = next_value if t == self.steps_per_batch - 1 else returns[t+1]
                    is_nonterminal = 1.0 - self.dones[t]
                    returns[t] = self.rewards[t] + self.gamma * is_nonterminal * next_return
                advantages = returns - self.values

        # Flatten the batch
        b_obs        = self.obs.reshape((-1, self.input_size))
        b_logprobs   = self.logprobs.reshape(-1)
        b_actions    = self.actions.reshape(-1)
        b_values     = self.values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)

        # Optimize the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

        # Update counters
        self.updates_done += 1
        self.current_step = 0

        # Log to Tensorboard
        if self.writer is not None:
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.total_steps_done)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps_done)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.total_steps_done)

def run_ppo(env, agents, seed, num_updates, steps_per_update, writer, device):
    fix_seeds(seed)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    time_step = env.reset()

    for update in range(num_updates):
        for step in range(steps_per_update):
            # Note: training loop assumes that each agent will always be called on to play before the game proceeds!
            # Ask each agent to pick an action
            for agent in agents:
                action = agent.step(time_step)
                time_step, reward, done = env.step(action.cpu().numpy(), reset_if_done=True)

            # Tell all agents their rewards and whether the episode is done
            for agent in agents:
                agent.post_step(reward, done)

        # Train
        for agent in agents:
            agent.learn(time_step)

        # Evaluate
        global_step = agents[0].total_steps_done
        if update % 1 == 0:
            logging.info("-" * 80)
            logging.info("Episode %s", global_step)
            avg_return = _eval_agent(
                Environment(pyspiel.load_game('catch'), chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=False, terminal_rewards=False), 
                agents[0], 
                100)
            logging.info("Avg return: %s", avg_return)
            
            writer.add_scalar("charts/avg_return", avg_return, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()

def main():
    setUpLogging()
    args = parse_args()
    run_name = f"{args.game_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Fix seed once here for network initialization
    fix_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    game = pyspiel.load_game(args.game_name)
    env = VectorEnv([
        Environment(game, chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=False, terminal_rewards=False) 
        for _ in range(args.num_envs)
    ])

    info_state_shape = env.observation_spec()["info_state"]
    info_state_size = np.array(info_state_shape).prod()
    num_updates = args.total_timesteps // args.batch_size

    agents = [PPO(
        info_state_size,
        game.num_distinct_actions(),
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
    )]

    run_ppo(env, agents, args.seed, num_updates, args.num_steps, writer, device)


if __name__ == "__main__":
    main()
