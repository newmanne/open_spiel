import numpy as np
from ..env_decorator import AuctionStatTrackingDecorator
from open_spiel.python.vector_env import SyncVectorEnv
import logging
import time
from open_spiel.python.examples.ubc_utils import pretty_time

logger = logging.getLogger(__name__)

def eval_agent(env, agent, num_episodes, writer=None):
    """Evaluates `agent` for `num_episodes`."""
    if isinstance(env, SyncVectorEnv):
        raise ValueError("Use the parallel function for SyncVectorEnv.")
    
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
    
def eval_agents_parallel(env, agents, num_episodes, flat_rewards=False):
    """Evaluates `agent` for `num_episodes`."""
    if isinstance(env, SyncVectorEnv):
        num_envs = len(env)
    else:
        raise ValueError("Use SyncVectorEnv.")

    alg_start_time = time.time()
    logging.info(f"Evaluating for {num_episodes} episodes")

    total_rewards = np.zeros((num_envs, len(agents)))
    episode_counter = num_envs
    time_step = env.reset()

    while episode_counter < num_episodes:
        for agent in agents:
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step, rewards, dones, _ = env.step(agent_output, reset_if_done=True)

        total_rewards += np.array(rewards)
        episode_counter += sum(dones)

    finished = np.zeros(num_envs, dtype=bool)
    while not np.all(finished):
        for agent in agents:
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step, rewards, dones, _ = env.step(agent_output, reset_if_done=True)

        total_rewards[~finished] += np.array(rewards)[~finished]
        finished |= dones

    rewards = total_rewards.sum(axis=0) / episode_counter

    eval_time = time.time() - alg_start_time
    logging.info(f"Eval took {pretty_time(eval_time)} episodes")

    return {
        'rewards': rewards,
        'walltime': eval_time,
    }

DEFAULT_NUM_SAMPLES = 100_000
DEFAULT_REPORT_FREQ = 5000
DEFAULT_SEED = 1234
DEFAULT_COMPUTE_EFFICIENCY = False

def run_eval(env_and_policy, num_samples, report_freq=DEFAULT_REPORT_FREQ, seed=DEFAULT_SEED, compute_efficiency=DEFAULT_COMPUTE_EFFICIENCY):
    checkpoint = eval_agents_parallel(env_and_policy.env, env_and_policy.agents, num_samples)
    
    d = []
    for e in env_and_policy.envs:
        while not isinstance(e, AuctionStatTrackingDecorator):
            e = e._env
            d.append(e.stats_dict())

    stats_dict = d[0]
    for other_dict in d[1:]:
        for k, v in other_dict.items():
            stats_dict[k] += v

    checkpoint.update(stats_dict)
    return checkpoint
