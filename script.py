import sys
sys.path.append('/apps/open_spiel/spinningup')
from spinup import ppo_pytorch as ppo
import spinup.algos.pytorch.ppo.core as core
from open_spiel.python.examples.ubc_utils import UBCChanceEventSampler
import torch
import gym

from gym.vector import SyncVectorEnv
from gym.envs.registration import register
register(
    id='gym_examples/OpenSpiel-v0',
    entry_point='gym_examples.envs:OpenSpielEnv',
    max_episode_steps=300,
)

def make_non_vector_env():
    spiel_env_args = dict(chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=True, terminal_rewards=True)
    env = gym.make('gym_examples/OpenSpiel-v0', spiel_game_args={'filename': 'large_game_2.json'}, spiel_env_args=spiel_env_args)
    return env

def make_env():
    def make_spiel_game():
        env = gym.make('gym_examples/OpenSpiel-v0', spiel_game_args={'filename': 'large_game_2.json'}, spiel_env_args=spiel_env_args)
        return env
    spiel_env_args = dict(chance_event_sampler=UBCChanceEventSampler(), all_simultaneous=True, terminal_rewards=True)
    envs = gym.vector.SyncVectorEnv([make_spiel_game] * 2)
    return envs


env_fn = make_non_vector_env

ac_kwargs = dict(hidden_sizes=[64,64])

logger_kwargs = dict(output_dir='./ppo', exp_name='experiment_name')

ppo(env_fn=env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)