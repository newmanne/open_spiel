    # Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from open_spiel.python.examples.ubc_utils import  fix_seeds, pretty_time
from open_spiel.python.algorithms.exploitability import best_response
from absl import logging
import time
from open_spiel.python.examples.env_and_policy import EnvAndPolicy
from open_spiel.python.examples.ppo_utils import EpisodeTimer, PPOTrainingLoop, make_ppo_agent

def run_br(env_and_policy: EnvAndPolicy, br_player: int, total_timesteps: int, config, report_freq=1000, result_saver=None, seed=1234, compute_exact_br=False, use_wandb=False):
    fix_seeds(seed) # This was probably done above (to deal with network initilization), but do it again for good measure
    alg_start_time = time.time()

    env, agents, game = env_and_policy.env, env_and_policy.agents, env_and_policy.game
    agents[br_player] = make_ppo_agent(br_player, config, game)
    policy = env_and_policy.make_policy(agents)

    # TRAINING PHASE
    def report_hook(update, total_steps):
        logging.info(f"Update {update}. Training for {pretty_time(time.time() - alg_start_time)}")
        if compute_exact_br:
            logging.info("Computing exact BR")
            br = best_response(game, policy, br_player)
            gap = br['best_response_value'] - br['on_policy_value']
            logging.info(f"Gap between BR and current strategy: {gap}")

    report_timer = EpisodeTimer(report_freq)
    trainer = PPOTrainingLoop(env_and_policy, total_timesteps, players_to_train=[br_player], report_timer=report_timer, use_wandb=use_wandb)
    trainer.add_report_hook(report_hook)
    trainer.training_loop()

    ### Save the best responding agent
    walltime_train = time.time() - alg_start_time
    checkpoint = {
      'br_player': br_player,
      'walltime': walltime_train,
      'agent': agents[br_player].save(),
      'config': config,
      'episode': total_timesteps
    }
    logging.info(f'Walltime: {pretty_time(walltime_train)}')

    if result_saver is not None:
        result_saver.save(checkpoint)

    logging.info('All done. Goodbye!')