from dataclasses import dataclass, field
from open_spiel.python import rl_environment
from open_spiel.python.examples.env_and_policy import EnvAndPolicy
from open_spiel.python.examples.ubc_decorators import TremblingAgentDecorator
from open_spiel.python.examples.ubc_utils import *
import numpy as np
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.dqn import DQN
from open_spiel.python.utils.replay_buffer import ReplayBuffer
import time
import logging
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.vector_env import SyncVectorEnv
from open_spiel.python.env_decorator import NormalizingEnvDecorator, AuctionStatTrackingDecorator, StateSavingEnvDecorator, PotentialShapingEnvDecorator, TrapEnvDecorator
from typing import Callable, List
from dataclasses import asdict
from open_spiel.python.env_decorator import AuctionStatTrackingDecorator
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.examples.cfr_utils import make_cfr_agent
from open_spiel.python.observation import make_observation

logger = logging.getLogger(__name__)

PPO_DEFAULTS = {
  'num_envs': 8,
  'steps_per_batch': 128,
  'num_minibatches': 4,
  'update_epochs': 4,
  'learning_rate': 2.5e-4,
  'anneal_lr': False,
  'gae': True,
  'gamma': 0.99,
  'gae_lambda': 0.95,
  'normalize_advantages': True,
  'use_returns_as_advantages': False,
  'clip_coef': 0.2,
  'clip_vloss': True,
  'agent_fn': 'PPOAgent',
  'agent_fn_kwargs': {},
  'entropy_coef': 0.01,
  'value_coef': 0.5,
  'max_grad_norm': 0.5,
  'target_kl': None,
  'device': default_device(),
  'use_wandb': False,
  'optimizer': 'adam',
  'optimizer_kwargs': {},
  'use_sos': False,
  'wall': False,
}

DQN_DEFAULTS = {
  'hidden_layers_sizes': 128,
  'replay_buffer_capacity': 10000,
  'batch_size': 128,
  'replay_buffer_class': ReplayBuffer,
  'learning_rate': 0.01,
  'update_target_network_every': 1000,
  'learn_every': 10,
  'discount_factor': 1.0,
  'min_buffer_size_to_learn': 1000,
  'epsilon_start': 1.0,
  'epsilon_end': 0.1,
  'epsilon_decay_duration': 1_000_000,
  'optimizer_str': "sgd",
  'loss_str': "mse",
  'agent_fn': 'mlp',
  'agent_fn_kwargs': {},
  'use_wandb': False,
}
    

def read_dqn_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    config = {**DQN_DEFAULTS, **config}  # priority from right to left

    print(config)
    return config



def read_ppo_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    config = {**PPO_DEFAULTS, **config}  # priority from right to left

    print(config)

    return config

# def make_schedule_function(func_name, max_t, initial_frac = 0.5):
#   if func_name == 'linear':
#     return lambda t: initial_frac * (1- (t/max_t))
#   elif func_name == 'constant':
#     return lambda t: initial_frac
#   else: 
#     raise NotImplementedError()

# def make_reward_function(func_name):
#   if func_name.startswith('neg_'):
#     reward_function = make_reward_function(func_name[4:])
#     return lambda state: -reward_function(state)
#   else:
#     def generic_reward(state):
#       attr = getattr(state, func_name)
#       if isinstance(attr, Callable):
#         return attr()
#       else:
#         return attr
#     return generic_reward


def make_potential_function(func_name):
  if '_potential_normalized' not in func_name:
    func_name += '_potential_normalized'
  if func_name.startswith('neg_'):
    reward_function = make_potential_function(func_name[4:])
    return lambda state: -reward_function(state)
  else:
    def generic_reward(state):
      return getattr(state, func_name)
      # attr = getattr(state, func_name)
      # if isinstance(attr, Callable):
      #   return attr()
      # else:
      #   return attr
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
  # reward_function: str = None
  # schedule_function: str = None
  # initial_frac: float = 0.5
  # total_timesteps: int = None

  potential_function: str =  None

  use_wandb: bool = False
  clear_on_report: bool = False
  observer_params: dict = None

  scale_coef: float = 1.

  trap_value: float = None
  trap_delay: int = 0

  include_state: bool = False


  def make_env(self, game):
    if not self.sync and self.num_envs > 1:
      raise ValueError("Sync must be True if num_envs > 1")
    
    # This is better, but still will use a new one each time you eval...
    observer = make_observation(game, params=self.observer_params)

    def gen_env(seed, env_id=0):
        # Only track env_id == 0 so we don't have multi-valued metrics
        env = rl_environment.Environment(game, chance_event_sampler=UBCChanceEventSampler(seed=seed), history_prefix=self.history_prefix, observer=observer, include_state=self.include_state)
        if self.num_states_to_save:
          logger.info("State saving decorator")
          env = StateSavingEnvDecorator(env, self.num_states_to_save)
        if self.track_stats:
          logger.info("Tracking stats decorator")
          env = AuctionStatTrackingDecorator(env, self.clear_on_report)
        if self.normalize_rewards:
          logger.info("Reward normalizing decorator")
          env = NormalizingEnvDecorator(env, reward_normalizer=torch.tensor(np.maximum(game.upper_bounds, game.lower_bounds)))
        if self.trap_value is not None: # Needs to happen after normalizing
          logger.info("Traps with penalty {} and delay {}".format(self.trap_value, self.trap_delay))
          env = TrapEnvDecorator(env, self.trap_value, self.trap_delay)
        if self.potential_function:
          potential_function = make_potential_function(self.potential_function)
          logger.info("Shaping potential with function: {} and scale strength {}".format(self.potential_function, self.scale_coef))
          env = PotentialShapingEnvDecorator(env, potential_function, game.num_players(), scale_coef=self.scale_coef)        
        return env
    
    if self.sync:
      env = SyncVectorEnv(
          [gen_env(self.seed + i, env_id=i) for i in range(self.num_envs)]
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

  def __init__(self, frequency, early_frequency=None, fixed_episodes=None, eval_zero=False, every_seconds=None):
    if fixed_episodes is None:
      fixed_episodes = []
    self.fixed_episodes = fixed_episodes

    self.frequency = frequency
    self.early_frequency = early_frequency
    self.cur_frequency = self.frequency if self.early_frequency is None else self.early_frequency
    self.eval_zero = eval_zero
    self.last_known_ep = -1

    self.every_seconds = every_seconds
    self.last_update_time = time.time()

  def should_trigger(self, ep):
    if ep > self.frequency: # Move on from early frequency if needed
      self.cur_frequency = self.frequency
    
    while ep > self.last_known_ep:
      self.last_known_ep += 1 
      if self._should_trigger(self.last_known_ep):
        # Note in the reports you might see unround numbers because of how we do it (e.g., logs for episode 10_007)
        self.last_known_ep = ep
        self.last_update_time = time.time()
        return True

    # Lastly, check if time expired
    if self.every_seconds and time.time() - self.last_update_time > self.every_seconds:
      self.last_update_time = time.time()
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

def make_dqn_kwargs_from_config(config):
  dqn_kwargs = {**DQN_DEFAULTS, **config}  # priority from right to left
  dqn_kwargs = {k:v for k,v in dqn_kwargs.items() if k in DQN_DEFAULTS.keys()} 
  return dqn_kwargs

def make_ppo_agent(player_id, config, game):
    num_players, num_actions, num_products = game.num_players(), game.num_distinct_actions(), game.auction_params.num_products

    # Double actions when using traps in the network
    if config.get('trap_value', None): 
      num_actions *= 2

    state_shape = rl_environment.Environment(game).observation_spec()["info_state_shape"]

    # TODO: Do you want to parameterize NN size/architecture?
    ppo_kwargs = make_ppo_kwargs_from_config(config)

    agent = PPO(
        input_shape=state_shape,
        num_actions=num_actions,
        num_players=num_players,
        player_id=player_id,
        **ppo_kwargs
    )

    if config.get('tremble', None):
      logger.info("Adding trembling agent decorator")
      agent = TremblingAgentDecorator(agent, config['tremble'])

    return agent

def make_dqn_agent(player_id, config, game):
    num_players, num_actions, num_products = game.num_players(), game.num_distinct_actions(), game.auction_params.num_products
    state_shape = rl_environment.Environment(game).observation_spec()["info_state_shape"]
    dqn_kwargs = make_dqn_kwargs_from_config(config)
    agent = DQN(
        player_id,
        state_shape,
        num_actions,
        **dqn_kwargs
    )
    return agent


def make_env_and_policy(game, config, env_params=None):
  if env_params is None:
    env_params = EnvParams(num_envs=config.get('num_envs', 1), seed=config.get('seed', 1234))

  solver_type = config.get('solver_type', 'ppo')
  if solver_type == 'cfr':
      agent_fn = make_cfr_agent
      env_params.include_state = True
      env_params.normalize_rewards = False
  elif solver_type == 'dqn':
      agent_fn = make_dqn_agent
  else:
      agent_fn = make_ppo_agent


  env = env_params.make_env(game)
  agents = [agent_fn(player_id, config, game) for player_id in range(game.num_players())]
  return EnvAndPolicy(env=env, agents=agents, game=game)

class PPOTrainingLoop:

  def __init__(self, game, env, agents, total_timesteps, players_to_train=None, report_timer=None, eval_timer=None, policy_diff_threshold=1e-3, max_policy_diff_count=9999, use_wandb=False, wandb_step_interval=1024):
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
    self.use_wandb = use_wandb
    self.wandb_step_interval = wandb_step_interval

  def add_report_hook(self, hook):
    self.report_hooks.append(hook)

  def add_eval_hook(self, hook):
    self.eval_hooks.append(hook)

  def training_loop(self):
    agent_zero = self.agents[self.players_to_train[0]] # Assuming it's all the same across agents...
    model = None
    if hasattr(agent_zero, 'steps_per_batch'):
      num_steps = agent_zero.steps_per_batch
      model = 'PPO'
    elif hasattr(agent_zero, '_learn_every'):
      num_steps = agent_zero._learn_every
      model = 'DQN'
    else:
      raise ValueError("I don't know what you are training")

    batch_size = int(len(self.env) * num_steps)
    num_updates = self.total_timesteps // batch_size
    if self.use_wandb:
      wandb_update_interval = max(self.wandb_step_interval // batch_size, 1)

    logging.info(f"Training for {num_updates} updates")
    logging.info(f"Fixed agents are {self.fixed_agents}. Learning agents are {self.players_to_train}")
    time_step = self.env.reset()

    for update in range(1, num_updates + 1):
      if self.report_timer is not None and self.report_timer.should_trigger(update * batch_size):
        for hook in self.report_hooks:
          hook(update, update * batch_size)
      if self.eval_timer is not None and self.eval_timer.should_trigger(update * batch_size):
        for hook in self.eval_hooks:
          hook(update, update * batch_size)

      for _ in range(num_steps):
          for player_id, agent in enumerate(self.agents): 
              
              agent_output = agent.step(time_step, is_evaluation=player_id in self.fixed_agents)
              # Could get round from TS here and log probs?
              time_step, reward, done, unreset_time_steps = self.env.step(agent_output, reset_if_done=True)

          if model == 'PPO':
            for player_id, agent in enumerate(self.agents):
              if player_id in self.players_to_train:  
                agent.post_step([r[player_id] for r in reward], done)

      if model == 'PPO':
        policy_changed = False
        for player_id, agent in enumerate(self.agents):
          if player_id in self.players_to_train:
            if agent.anneal_lr:
              agent.anneal_learning_rate(update - 1, num_updates)
            agent.learn(time_step)
          if agent.get_max_policy_diff() >= self.policy_diff_threshold:
            policy_changed = True

      # Commit wandb
      if self.use_wandb and update % wandb_update_interval == 0:
        # TODO: Make this way less specific to our game/abstract it more
        import wandb
        stats_dict = AuctionStatTrackingDecorator.merge_stats(self.env)
        log_stats_dict = dict()
        prefix = 'metrics'
        if 'revenues' in stats_dict:
            log_stats_dict[f'{prefix}/mean_revenue'] = np.mean(stats_dict['revenues'])
        if 'auction_lengths' in stats_dict:
            log_stats_dict[f'{prefix}/mean_auction_length'] = np.mean(stats_dict['auction_lengths'])
        if 'welfares' in stats_dict:
            log_stats_dict[f'{prefix}/mean_welfare'] = np.mean(stats_dict['welfares'])
       
        for player_id in range(len(self.agents)):
          if 'raw_rewards' in stats_dict:
            log_stats_dict[f'{prefix}/player_{player_id}_mean_reward'] = np.mean(stats_dict['raw_rewards'][player_id])
          if 'payments' in stats_dict:
            log_stats_dict[f'{prefix}/player_{player_id}_payment'] = np.mean(stats_dict['payments'][player_id])
            # f'player_{player_id}_allocation': self.allocations[player_id][-1], # TODO? Probably needs to be per product

          if 'traps' in stats_dict:
            log_stats_dict[f'{prefix}/player_{player_id}_traps'] = np.mean(stats_dict['traps'][player_id])
 
        # Initial state action probabilities -- disabled for now
        # TODO: DONT MAKE THIS EACH TIME, JUST RESET IT 
        # e = EnvParams(num_envs=1, sync=False, normalize_rewards=False).make_env(self.game)
        # e.reset()
        # for player_id in range(len(self.agents)):
        #   step_output = self.agents[player_id].step(e.get_time_step(), is_evaluation=True)
        #   probs = step_output.probs
        #   e.step([step_output.action])
        #   for action_id in range(len(probs)):
        #     log_stats_dict[f'actions/player_{player_id}_action_{action_id}_prob'] = probs[action_id]
        

          # from open_spiel.python.observation import make_observation
          # observer = make_observation(self.game, params=observer_params)
          # initial_state = self.game.new_initial_state()
          # observer.set_from(initial_state, player=player_id)
          # info_state = np.array(observer.tensor)
          # probs = self.agents[player_id].network.get_action_and_value(torch.tensor(info_state)).probs
          # for action_id in range(len(probs)):
          #   log_stats_dict[f'{prefix}/player_{player_id}_action_{action_id}_prob'] = probs[action_id]

        # TODO:
        # for player_id in range(self.n_players):
        #     metrics[f'metrics/player_{player_id}_unshaped_reward'] = time_step.rewards[player_id]
        #     metrics[f'metrics/player_{player_id}_shaped_reward'] = new_rewards[player_id] - time_step.rewards[player_id]

        # This should be the ONLY commit=True. Step sizes will now be in terms of updates
        wandb.log(log_stats_dict, commit=True)

      if model == 'PPO':
        if not policy_changed:
          self.policy_diff_count += 1
          if self.policy_diff_count >= self.max_policy_diff_count:
            logging.info("Policy has not changed for {} updates. Stopping training".format(self.max_policy_diff_count))
            break
        else:
          self.policy_diff_count = 0  


    logging.info(f"Terminating {model} training after {update} updates and {update * batch_size} steps")
    update += 1 # Prevent stupid DB mismatch errors
    for hook in self.report_hooks:
      hook(update, update * batch_size)
    for hook in self.eval_hooks:
      hook(update, update * batch_size)


def ppo_checkpoint(env_and_model, step, alg_start_time, compute_nash_conv=False, update=None):
  if compute_nash_conv:
    raise ValueError("Nash conv not supported for PPO")

  msg = f"EVALUATION AFTER {step} steps"
  if update is not None:
    msg += f" (update {update})"
  logger.info(msg)

  policy = env_and_model.make_policy()
  checkpoint = {
      'walltime': time.time() - alg_start_time,
      'policy': policy.save(),
      'episode': step,
  }
  return checkpoint

def run_ppo(env_and_policy, total_steps, result_saver=None, seed=1234, compute_nash_conv=False, dispatcher=None, report_timer=None, eval_timer=None, use_wandb=False, wandb_step_interval=1024):
  # This may have already been done, but do it again. Required to do it outside to ensure that networks get initilized the same way, which usually happens elsewhere
  fix_seeds(seed)
  game, env, agents = env_and_policy.game, env_and_policy.env, env_and_policy.agents

  alg_start_time = time.time()
  trainer = PPOTrainingLoop(game, env, agents, total_steps, report_timer=report_timer, eval_timer=eval_timer, use_wandb=use_wandb, wandb_step_interval=wandb_step_interval)

  def eval_hook(update, total_steps):
    logging.info("Running eval hook")
    checkpoint = ppo_checkpoint(env_and_policy, total_steps, alg_start_time, compute_nash_conv=compute_nash_conv, update=update)
    if result_saver is not None:
      checkpoint_name = result_saver.save(checkpoint)
      if dispatcher is not None:
        dispatcher.dispatch(checkpoint_name)
    logging.info("Done eval hook")
  
  def report_hook(update, total_steps):
    logging.info(f"Update {update}")

  trainer.add_report_hook(report_hook)
  trainer.add_eval_hook(eval_hook)
  trainer.training_loop()

  logging.info(f"Walltime: {pretty_time(time.time() - alg_start_time)}")
  logging.info('All done. Goodbye!')
