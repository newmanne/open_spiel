from builtins import classmethod
from open_spiel.python.rl_environment import Environment
import torch
from collections import defaultdict
import numpy as np
import collections
from typing import Callable

class EnvDecorator(object):

    def __init__(self, env: Environment) -> None:
        self._env = env
        self.env_attributes = [attribute for attribute in self._env.__dict__.keys()]
        self.env_methods = [m for m in dir(self._env) if not m.startswith('_') and m not in self.env_attributes]

    def __getattr__(self, func):
        if func in self.env_methods:
            def method(*args):
                return getattr(self._env, func)(*args)
            return method
        elif func in self.env_attributes:
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

    def __init__(self, env: Environment, reward_normalizer: torch.tensor = None) -> None:
        super().__init__(env)
        self.reward_normalizer = reward_normalizer
    
    def get_time_step(self):
        time_step = self._env.get_time_step()

        if self.reward_normalizer is not None:
            time_step.rewards[:] = (torch.tensor(time_step.rewards) / self.reward_normalizer).tolist() 

        if np.isnan(time_step.rewards).any():
            raise ValueError("Nan reward after normalization!")
        return time_step


class PotentialShapingEnvDecorator(EnvDecorator):

    def __init__(self, env: Environment, potential_function: Callable, use_wandb: bool) -> None:
        super().__init__(env)
        self.potential_function = potential_function
        self.n_players = 2
        self.last_potential = np.zeros(self.n_players)
        self.use_wandb = use_wandb
    
    def get_time_step(self):
        time_step = self._env.get_time_step()
        state = self._env._state
        if time_step.last():
            # There's no reason to add the current potential just to subtract it - it won't make a difference. So let's just undo all the potentials right now
            shaped_reward = -self.last_potential
        else:
            current_potential = self.potential_function(state)
            shaped_reward = current_potential - self.last_potential
            self.last_potential = current_potential

        new_rewards = torch.tensor(time_step.rewards) + shaped_reward

        if self.use_wandb:
            import wandb
            metrics = dict()
            for player_id in range(self.n_players):
                metrics[f'metrics/player_{player_id}_unshaped_reward'] = time_step.rewards[player_id]
                metrics[f'metrics/player_{player_id}_shaped_reward'] = new_rewards[player_id] - time_step.rewards[player_id]
            wandb.log(metrics, commit=False)

        time_step.rewards[:] = new_rewards
        return time_step

    def reset(self):
        _ = self._env.reset()
        self.last_potential = np.zeros(self.n_players)
        return self.get_time_step()

class RewardShapingEnvDecorator(EnvDecorator):

    def __init__(self, env: Environment, reward_function: Callable, schedule_function: Callable) -> None:
        super().__init__(env)
        self.reward_function = reward_function
        self.schedule_function = schedule_function
        self.t = 0
    
    def get_time_step(self):
        time_step = self._env.get_time_step()
        state = self._env._state
        lam = self.schedule_function(self.t)
        new_rewards = lam * self.reward_function(state) + (1 - lam) * torch.tensor(time_step.rewards)
        time_step.rewards[:] = new_rewards
        return time_step

    def step(self, step_outputs):
        _ = self._env.step(step_outputs)
        self.t += 1
        return self.get_time_step()

class StateSavingEnvDecorator(EnvDecorator):

    def __init__(self, env: Environment, num_states_to_save = 100) -> None:
        super().__init__(env)
        self.num_states_to_save = num_states_to_save
        self.num_players = env.num_players
        self.states = [collections.deque() for _ in range(self.num_players)]
    
    def step(self, step_outputs):
        time_step = self._env.get_time_step()
        if not time_step.last():
            current_player = time_step.current_player()
            self.states[current_player].append(time_step.observations['info_state'][current_player])

        _ = self._env.step(step_outputs)
        return time_step

    def get_states(self):
        return [list(d) for d in self.states]

    @staticmethod
    def merge_states(sync_env):
        d = []
        for e in sync_env.envs:
            d += e.get_states()
        return d


class AuctionStatTrackingDecorator(EnvDecorator):

    def __init__(self, env: Environment, use_wandb: bool = False) -> None:
        super().__init__(env)
        self.rewards = defaultdict(list)
        self.payments = defaultdict(list)
        self.allocations = defaultdict(list)
        self.auction_lengths = []
        self.welfares = []
        self.revenues = []

        self.use_wandb = use_wandb

    def step(self, step_outputs):
        _ = self._env.step(step_outputs)
        time_step = self._env.get_time_step()
        state = self._env._state

        if time_step.last():
            for player_id, reward in enumerate(time_step.rewards):
                self.rewards[player_id].append(reward)
            for player_id, payment in enumerate(state.get_final_payments()):
                self.payments[player_id].append(payment)
            self.revenues.append(state.revenue)
            for player_id, allocation in enumerate(state.get_allocation()):
                self.allocations[player_id].append(allocation.tolist())
            self.auction_lengths.append(state.round)
            self.welfares.append(state.get_welfare())

            if self.use_wandb:
                import wandb
                metrics = {
                    'metrics/revenue': self.revenues[-1],
                    'metrics/auction_length': self.auction_lengths[-1],
                    'metrics/welfare': self.welfares[-1],
                }
                for player_id in range(self._env.num_players):
                    metrics[f'metrics/player_{player_id}_reward'] = self.rewards[player_id][-1]
                    metrics[f'metrics/player_{player_id}_payment'] = self.payments[player_id][-1]
                    # f'player_{player_id}_allocation': self.allocations[player_id][-1], # TODO? Probably needs to be per product
                wandb.log(metrics, commit=False)

        return self.get_time_step()

    def stats_dict(self):
        return {
            'raw_rewards': self.rewards,
            'allocations': self.allocations,
            'payments': self.payments,
            'auction_lengths': self.auction_lengths,
            'revenues': self.revenues,
            'welfares': self.welfares,
        }

    @staticmethod
    def merge_stats(sync_env):
        d = []
        for e in sync_env.envs:
            d.append(e.stats_dict())

        stats_dict = d[0]
        for other_dict in d[1:]:
            for k, v in other_dict.items():
                if isinstance(v, collections.defaultdict):
                    for k2, v2 in v.items():
                        stats_dict[k][k2] += v2
                else:
                    stats_dict[k] += v
        return stats_dict

