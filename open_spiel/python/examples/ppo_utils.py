from dataclasses import dataclass, field
from open_spiel.python import rl_environment
from open_spiel.python.examples.env_and_policy import EnvAndPolicy
from open_spiel.python.examples.ubc_utils import *
import numpy as np
import pandas as pd
from open_spiel.python.pytorch.ppo import PPO
import time
import logging
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.env_decorator import NormalizingEnvDecorator, AuctionStatTrackingDecorator, RewardShapingEnvDecorator, StateSavingEnvDecorator
from typing import Callable, List
from dataclasses import asdict


logger = logging.getLogger(__name__)

PPO_DEFAULTS = {
  'num_envs': 8,
  'steps_per_batch': 128,
  'num_minibatches': 4,
  'update_epochs': 4,
  'learning_rate': 2.5e-4,
  'num_annealing_updates': None,
  'gae': True,
  'gamma': 0.99,
  'gae_lambda': 0.95,
  'normalize_advantages': True,
  'clip_coef': 0.2,
  'clip_vloss': True,
  'agent_fn': 'PPOAgent',
  'entropy_coef': 0.01,
  'value_coef': 0.5,
  'max_grad_norm': 0.5,
  'target_kl': None,
  'device': default_device(),
  'use_wandb': False,
}

def read_ppo_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    config = {**PPO_DEFAULTS, **config}  # priority from right to left

    return config

def make_schedule_function(func_name, max_t, initial_frac = 0.5):
  if func_name == 'linear':
    return lambda t: initial_frac * (1- (t/max_t))
  elif func_name == 'constant':
    return lambda t: initial_frac
  else: 
    raise NotImplementedError()

def make_reward_function(func_name):
  if func_name.startswith('neg_'):
    reward_function = make_reward_function(func_name[4:])
    return lambda state: -reward_function(state)
  else:
    def generic_reward(state):
      attr = getattr(state, func_name)
      if isinstance(attr, Callable):
        return attr()
      else:
        return attr
    return generic_reward

@dataclass
class EnvParams:

  num_envs: int = 8
  normalize_rewards: bool = True
  seed: int = 1234
  track_stats: bool = False
  sync: bool = True
  history_prefix: List = field(default_factory=lambda: [])
  num_states_to_save: int = 0

  # Stuff related to reward shaping
  reward_function: str = None
  schedule_function: str = None
  initial_frac: float = 0.5
  total_timesteps: int = None

  use_wandb: bool = False

  def make_env(self, game):
    if not self.sync and self.num_envs > 1:
      raise ValueError("Sync must be True if num_envs > 1")

    def gen_env(seed):
        env = rl_environment.Environment(game, chance_event_sampler=UBCChanceEventSampler(seed=seed), use_observer_api=True, history_prefix=self.history_prefix)
        if self.num_states_to_save:
          env = StateSavingEnvDecorator(env, self.num_states_to_save)
        if self.track_stats:
          env = AuctionStatTrackingDecorator(env, self.use_wandb)
        if self.normalize_rewards:
          env = NormalizingEnvDecorator(env, reward_normalizer=torch.tensor(np.maximum(game.upper_bounds, game.lower_bounds)))
        if self.reward_function is not None or self.schedule_function is not None:
          if self.reward_function is None or self.schedule_function is None:
            raise ValueError("Must specify both reward_function and schedule_function")
          reward_function = make_reward_function(self.reward_function)
          schedule_function = make_schedule_function(self.schedule_function, self.total_timesteps / self.num_envs, self.initial_frac)
          env = RewardShapingEnvDecorator(env, reward_function, schedule_function)
        return env
    
    if self.sync:
      env = SyncVectorEnv(
          [gen_env(self.seed + i) for i in range(self.num_envs)]
      )
    else:
      env = gen_env(self.seed)
    return env

  @staticmethod
  def from_config(config):
    ## Config is a dict of params you want to override
    defaults = asdict(EnvParams())
    env_config = {k:v for k,v in config.items() if k in defaults}
    return EnvParams(**{**defaults, **env_config})

class EpisodeTimer:

  def __init__(self, frequency, early_frequency=None, fixed_episodes=None, eval_zero=False):
    if fixed_episodes is None:
      fixed_episodes = []
    self.fixed_episodes = fixed_episodes

    self.frequency = frequency
    self.early_frequency = early_frequency
    self.cur_frequency = self.frequency if self.early_frequency is None else self.early_frequency
    self.eval_zero = eval_zero
    self.last_known_ep = -1

  def should_trigger(self, ep):
    if ep > self.frequency:
      self.cur_frequency = self.frequency
    
    while ep > self.last_known_ep:
      self.last_known_ep += 1 
      if self._should_trigger(self.last_known_ep):
        # Note in the reports you might see unround numbers because of how we do it (e.g., logs for episode 10_007)
        self.last_known_ep = ep
        return True

    return False

  def _should_trigger(self, ep):
    return (ep > 0 and ep % self.cur_frequency == 0) or \
      ep in self.fixed_episodes or\
      ep == 0 and self.eval_zero


def make_ppo_kwargs_from_config(config):
  ppo_kwargs = {**PPO_DEFAULTS, **config}  # priority from right to left
  ppo_kwargs = {k:v for k,v in ppo_kwargs.items() if k in PPO_DEFAULTS.keys()} 
  return ppo_kwargs

def make_ppo_agent(player_id, config, game):
    num_players, num_actions, num_products = game.num_players(), game.num_distinct_actions(), game.auction_params.num_products

    state_size = rl_environment.Environment(game).observation_spec()["info_state"]

    # TODO: Do you want to parameterize NN size/architecture?
    ppo_kwargs = make_ppo_kwargs_from_config(config)

    return PPO(
        input_shape=state_size,
        num_actions=num_actions,
        num_players=num_players,
        player_id=player_id,
        **ppo_kwargs
    )

def make_env_and_policy(game, config, env_params=None):
  if env_params is None:
    env_params = EnvParams(num_envs=config['num_envs'], seed=config['seed'])
  env = env_params.make_env(game)
  agents = [make_ppo_agent(player_id, config, game) for player_id in range(game.num_players())]
  return EnvAndPolicy(env=env, agents=agents, game=game)

class PPOTrainingLoop:

  def __init__(self, game, env, agents, total_timesteps, players_to_train=None, report_timer=None, eval_timer=None, policy_diff_threshold=1e-3, max_policy_diff_count=5):
    self.game = game
    self.env = env
    self.agents = agents
    self.total_timesteps = total_timesteps
    self.players_to_train = players_to_train if players_to_train is not None else list(range(game.num_players()))
    self.fixed_agents = set(range(game.num_players())) - set(self.players_to_train)
    self.report_timer = report_timer     
    self.report_hooks = []
    self.eval_timer = eval_timer
    self.eval_hooks = []
    self.policy_diff_threshold = policy_diff_threshold
    self.max_policy_diff_count = max_policy_diff_count
    self.policy_diff_count = 0

  def add_report_hook(self, hook):
    self.report_hooks.append(hook)

  def add_eval_hook(self, hook):
    self.eval_hooks.append(hook)

  def training_loop(self):
    num_steps = self.agents[self.players_to_train[0]].steps_per_batch # Assuming it's all the same across agents...
    batch_size = int(len(self.env) * num_steps)
    num_updates = self.total_timesteps // batch_size

    logging.info(f"Training for {num_updates} updates")
    logging.info(f"Fixed agents are {self.fixed_agents}. Learning agents are {self.players_to_train}")
    for update in range(1, num_updates + 1):
      if self.report_timer is not None and self.report_timer.should_trigger(update):
        for hook in self.report_hooks:
          hook(update, update * num_steps)
      if self.eval_timer is not None and self.eval_timer.should_trigger(update):
        for hook in self.eval_hooks:
          hook(update, update * num_steps)

      time_step = self.env.reset()
      for _ in range(num_steps):
          for player_id, agent in enumerate(self.agents): 
              agent_output = agent.step(time_step, is_evaluation=player_id in self.fixed_agents)
              time_step, reward, done, unreset_time_steps = self.env.step(agent_output, reset_if_done=True)

          for player_id, agent in enumerate(self.agents):
            if player_id in self.players_to_train:  
              agent.post_step([r[player_id] for r in reward], done)

      policy_changed = False
      for player_id, agent in enumerate(self.agents):
        if player_id in self.players_to_train:
          agent.learn(time_step)
        if agent.get_max_policy_diff() >= self.policy_diff_threshold:
          policy_changed = True
      
      if not policy_changed:
        self.policy_diff_count += 1
        if self.policy_diff_count >= self.max_policy_diff_count:
          logging.info("Policy has not changed for {} updates. Stopping training".format(self.max_policy_diff_count))
          break
      else:
        self.policy_diff_count = 0  


    logging.info(f"Terminating PPO training after {update} updates")

    # Lastly, call all hooks
    for hook in self.report_hooks:
      hook(update, update * num_steps)
    for hook in self.eval_hooks:
      hook(update, update * num_steps)


def ppo_checkpoint(env_and_model, step, alg_start_time, compute_nash_conv=False):
    logger.info(f"EVALUATION AFTER {step} steps")

    policy = env_and_model.make_policy()
    if compute_nash_conv:
        logging.info('Computing nash conv...')
        n_conv = nash_conv(env_and_model.game, policy, use_cpp_br=True)
        logging.info(f"{n_conv}")
        logging.info("_____________________________________________")
    else:
        n_conv = None

    checkpoint = {
        'walltime': time.time() - alg_start_time,
        'policy': policy.save(),
        'nash_conv_history': [],
        'episode': step,
    }
    return checkpoint

def run_ppo(env_and_policy, total_steps, result_saver=None, seed=1234, compute_nash_conv=False, dispatcher=None, report_timer=None, eval_timer=None):
  # This may have already been done, but do it again. Required to do it outside to ensure that networks get initilized the same way, which usually happens elsewhere
  fix_seeds(seed)
  game, env, agents = env_and_policy.game, env_and_policy.env, env_and_policy.agents

  alg_start_time = time.time()
  trainer = PPOTrainingLoop(game, env, agents, total_steps, report_timer=report_timer, eval_timer=eval_timer)

  def eval_hook(update, total_steps):
    checkpoint = ppo_checkpoint(env_and_policy, total_steps, alg_start_time, compute_nash_conv=compute_nash_conv)
    if result_saver is not None:
      checkpoint_name = result_saver.save(checkpoint)
      if dispatcher is not None:
        dispatcher.dispatch(checkpoint_name)
  
  def report_hook(update, total_steps):
    logging.info(f"Update {update}")

  trainer.add_report_hook(report_hook)
  trainer.add_eval_hook(eval_hook)
  trainer.training_loop()

  logging.info(f"Walltime: {pretty_time(time.time() - alg_start_time)}")
  logging.info('All done. Goodbye!')
