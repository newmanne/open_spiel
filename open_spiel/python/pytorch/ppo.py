# Note: code adapted (with permission) from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

INVALID_ACTION_PENALTY = -1e6
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

def string_to_activation(activation_string):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }
    try:
        return activations[activation_string.lower()]
    except KeyError:
        raise ValueError(f'Invalid activation {activation_string}; valid activations are {list(activations.keys())}')

def build_sequential_network(observation_shape, output_size, hidden_sizes, activation, final_std):
    layers = []
    layers.append(layer_init(nn.Linear(np.array(observation_shape).prod(), hidden_sizes[0])))
    layers.append(activation())
    for i in range(len(hidden_sizes) - 1):
        layers.append(layer_init(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
        layers.append(activation())
    layers.append(layer_init(nn.Linear(hidden_sizes[-1], output_size), std=final_std))
    return nn.Sequential(*layers)

class PPOAgent(nn.Module):
    def __init__(self, num_actions, observation_shape, device, actor_hidden_sizes=None, actor_activation=nn.Tanh, critic_hidden_sizes=None, critic_activation=nn.Tanh):
        super().__init__()

        if actor_hidden_sizes is None:
            actor_hidden_sizes = [64, 64]
        if critic_hidden_sizes is None:
            critic_hidden_sizes = [64, 64]

        if isinstance(actor_activation, str):
            actor_activation = string_to_activation(actor_activation)
        if isinstance(critic_activation, str):
            critic_activation = string_to_activation(critic_activation)

        # Construct networks
        self.critic = build_sequential_network(observation_shape, 1, critic_hidden_sizes, critic_activation, final_std=1.0)
        self.actor = build_sequential_network(observation_shape, num_actions, actor_hidden_sizes, actor_activation, final_std=0.01)

        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits = self.actor(x)
        if torch.isnan(logits).any():
            raise ValueError("Training is messed up - logits are NaN")

        probs = CategoricalMasked(logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs.probs


class PPOAtariAgent(nn.Module):
    def __init__(self, num_actions, observation_shape, device):
        super(PPOAtariAgent, self).__init__()
        # Note: this network is intended for atari games, taken from https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
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
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()
        
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
            
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), probs.probs

def legal_actions_to_mask(legal_actions_list, num_actions):
    '''Convert a list of legal actions to a mask of size num actions with a 1 in a legal position'''
    legal_actions_mask = torch.zeros((len(legal_actions_list), num_actions), dtype=torch.bool)
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask

class PPO(nn.Module):
    """PPO Agent implementation in PyTorch.

    See open_spiel/python/examples/ppo_example.py for an usage example.

    Note that PPO runs multiple environments concurrently on each step (see 
    open_spiel/python/vector_env.py). In practice, this tends to improve PPO's
    performance. The number of parallel environments is controlled by the 
    num_envs argument. 
    """
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
        writer=None, # Tensorboard SummaryWriter
        use_wandb=True,
        agent_fn=PPOAtariAgent,
        agent_fn_kwargs=None,
        ):
        super().__init__()

        if agent_fn_kwargs is None:
            agent_fn_kwargs = {}

        if isinstance(agent_fn, str):
            if agent_fn == 'PPOAgent':
                agent_fn = PPOAgent
            else:
                raise ValueError(f"Unknown agent_fn {agent_fn}")

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
        self.use_wandb = use_wandb
        self.watch_output = None

        # Initialize networks
        self.network = agent_fn(self.num_actions, self.input_shape, device, **agent_fn_kwargs).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)
        
        # Initialize training buffers
        self.legal_actions_mask = torch.zeros((self.steps_per_batch, self.num_envs, self.num_actions), dtype=torch.bool).to(device)
        self.obs = torch.zeros((self.steps_per_batch, self.num_envs) + self.input_shape).to(device)
        self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.probs = torch.zeros((self.steps_per_batch, self.num_envs, self.num_actions)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)

        # Initialize counters 
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()

        self.max_policy_diff = 9999

        self._savers = [
            ("ppo_network", self.network),
        ]

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
                probs = probs.cpu().numpy()
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
                self.probs[self.cur_batch_idx] = probs
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
        if self.use_wandb and self.watch_output is None:
            pass # Honestly, this hasn't been super useful and takes up a lot of log bandwidth, but uncomment to watch the weights in wandb
            # import wandb
            # wandb.watch(self, log_freq=5)

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
        b_legal_actions_mask = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_probs = self.probs.reshape((-1, self.num_actions))
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

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(b_obs[mb_inds], legal_actions_mask=b_legal_actions_mask[mb_inds], action=b_actions.long()[mb_inds])
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

        # compute L_\infty change in probabilities
        # TODO: split into minibatches?
        _, _, _, _, new_probs = self.get_action_and_value(b_obs, legal_actions_mask=b_legal_actions_mask, action=b_actions.long())
        self.max_policy_diff = (new_probs - b_probs).abs().max().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if self.writer is not None:
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.total_steps_done)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps_done)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps_done)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps_done)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.total_steps_done)
            self.writer.add_scalar("charts/SPS", int(self.total_steps_done / (time.time() - self.start_time)), self.total_steps_done)

        if self.use_wandb:
            import wandb
            prefix = f'network_{self.player_id}'
            wandb.log({
                f"{prefix}/learning_rate": self.optimizer.param_groups[0]["lr"],
                f"{prefix}/value_loss": v_loss.item(),
                f"{prefix}/policy_loss": pg_loss.item(),
                f"{prefix}/entropy": entropy_loss.item(),
                f"{prefix}/old_approx_kl": old_approx_kl.item(),
                f"{prefix}/approx_kl": approx_kl.item(),
                f"{prefix}/clipfrac": np.mean(clipfracs),
                f"{prefix}/explained_variance": explained_var,
                f"{prefix}/SPS": int(self.total_steps_done / (time.time() - self.start_time)), # TODO: This is a bit silly if you do anything like an inline eval
                f"{prefix}/total_steps_done": self.total_steps_done,
                f"{prefix}/updates_done": self.updates_done,
                f"{prefix}/max_policy_diff": self.max_policy_diff,
            }, commit=False)

        # Update counters 
        self.updates_done += 1
        self.cur_batch_idx = 0

    def save(self):
        restore_dict = dict()
        for name, model in self._savers:
            restore_dict[name] = model.state_dict()
        return restore_dict

    def restore(self, restore_dict):
        for name, model in self._savers:
            model.load_state_dict(restore_dict[name])

    def get_max_policy_diff(self):
        return self.max_policy_diff
        