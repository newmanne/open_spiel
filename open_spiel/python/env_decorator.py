from open_spiel.python.rl_environment import Environment
import torch
from collections import defaultdict
import numpy as np

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

class AuctionStatTrackingDecorator(EnvDecorator):

    def __init__(self, env: Environment) -> None:
        super().__init__(env)
        self.rewards = defaultdict(list)
        self.payments = defaultdict(list)
        self.allocations = defaultdict(list)
        self.auction_lengths = []

    def get_time_step(self):
        time_step = self._env.get_time_step()
        state = self._env._state
        
        if time_step.last():
            for player_id, reward in enumerate(time_step.rewards):
                self.rewards[player_id].append(reward)
            for player_id, payment in enumerate(state.get_final_payments()):
                self.payments[player_id].append(payment)
            for player_id, allocation in enumerate(state.get_allocation()):
                self.allocations[player_id].append(allocation.tolist())
            self.auction_lengths.append(state.round)

        # TODO: Prices, types, efficiency

        return time_step

    def stats_dict(self):
        return {
            'raw_rewards': self.rewards,
            'allocations': self.allocations,
            'payments': self.payments,
            'auction_lengths': self.auction_lengths,
        }
