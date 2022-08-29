import numpy as np

from .ppo_utils import EpisodeTimer
from ..env_decorator import AuctionStatTrackingDecorator
from open_spiel.python.vector_env import SyncVectorEnv
import logging
import time
from open_spiel.python.examples.ubc_utils import pretty_time
import collections

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
    
def eval_agents_parallel(env, agents, num_episodes, report_timer=None, flat_rewards=False):
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
        if report_timer is not None and report_timer.should_trigger(episode_counter):
            # TODO: This logging only works sporadically because episode counter can increment by more than 1 - how do we not miss logs?
            logging.info(f"Evaluated {episode_counter}/{num_episodes} episodes")

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

        # Note: This might mean the decoartor for auction stats still adds episodes we don't really want to bias
        total_rewards[~finished] += np.array(rewards)[~finished]
        finished |= dones

    rewards = total_rewards.sum(axis=0) / episode_counter

    eval_time = time.time() - alg_start_time
    logging.info(f"Eval took {pretty_time(eval_time)} for the episodes")

    return {
        'rewards': rewards, # These will probably be normalized
        'walltime': eval_time,
    }

class EvalDefaults:
    DEFAULT_NUM_SAMPLES = 100_000
    DEFAULT_REPORT_FREQ = 5000
    DEFAULT_SEED = 1234
    DEFAULT_COMPUTE_EFFICIENCY = False

def run_eval(env_and_policy, num_samples=EvalDefaults.DEFAULT_NUM_SAMPLES, report_freq=EvalDefaults.DEFAULT_REPORT_FREQ, seed=EvalDefaults.DEFAULT_SEED, compute_efficiency=EvalDefaults.DEFAULT_COMPUTE_EFFICIENCY):
    # TODO: So many unused vars here
    checkpoint = eval_agents_parallel(env_and_policy.env, env_and_policy.agents, num_samples, report_timer=EpisodeTimer(report_freq))
    
    # TODO: Move to a classmethod on AuctionStatTrackingDecorator
    d = []
    for e in env_and_policy.env.envs:
        d.append(e.stats_dict())
        print(e.stats_dict())

    stats_dict = d[0]
    for other_dict in d[1:]:
        for k, v in other_dict.items():
            if isinstance(v, collections.defaultdict):
                for k2, v2 in v.items():
                    stats_dict[k][k2] += v2
            else:
                stats_dict[k] += v


    checkpoint.update(stats_dict)
    return checkpoint
