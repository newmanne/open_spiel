from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_utils import run_ppo, EpisodeTimer, read_ppo_config
from open_spiel.python.examples.ubc_utils import fix_seeds, apply_optional_overrides, default_device, setup_directory_structure
import sys
import logging
from auctions.models import *
from auctions.webutils import *
from auctions.savers import DBBRDispatcher, DBPolicySaver
import json
from distutils import util
from open_spiel.python.examples.ubc_dispatch import dispatch_experiments, dispatch_eval_database, dispatch_from_checkpoint
import time
from open_spiel.python.pytorch.mutations import mutate_sm
from open_spiel.python.examples.ppo_eval import eval_agents_parallel, run_eval
from open_spiel.python.env_decorator import StateSavingEnvDecorator, AuctionStatTrackingDecorator
import copy

logger = logging.getLogger(__name__)

METRICS = {
    'welfare': lambda samples: np.mean(samples['welfare']),
    'revenue': lambda samples: np.mean(samples['revenue']),
    'auction_length': lambda samples: np.mean(samples['auction_length']),
}


def parent_to_mutation(env_and_policy, num_states_to_save=1000, **mutation_kwargs):
    # Step 0: Copy old env and policy
    env_and_policy = copy.deepcopy(env_and_policy)

    # Step 1: Get a set of states
    tracking_env = EnvParams(num_states_to_save=num_states_to_save).make_env(env_and_policy.game)
    agents = env_and_policy.agents
    eval_agents_parallel(tracking_env, agents, num_episodes=num_states_to_save) # TODO: This is way too many states
    states = StateSavingEnvDecorator.merge_states(tracking_env)
    for player_id in range(len(agents)):
        model = env_and_policy.agents[player_id].network.actor
        # Step 2: Mutate the model
        new_model = mutate_sm(model, states[player_id], **mutation_kwargs)
        env_and_policy.agents[player_id].network.actor = new_model
    return env_and_policy

class Command(BaseCommand):
    help = 'Evolutionary algorithms!'

    def add_arguments(self, parser):
        # Naming
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--filename', type=str, default='parameters') # Select clock auction game
        parser.add_argument('--game_name', type=str, default='python_clock_auction')
        parser.add_argument('--network_config_file', type=str, default='aug11')

        # Evolution
        parser.add_argument('--population_size', type=int, default=5)
        parser.add_argument('--candidate_mutant_size', type=int, default=100)
        parser.add_argument('--max_generations', type=int, default=10)
        parser.add_argument('--fitness_function', type=str, default="welfare")
        parser.add_argument('--poll_interval', type=float, default=30.)
        parser.add_argument('--sm_states', type=int, default=1000)
        parser.add_argument('--mutation_strength', type=float, default=2e-2)
        parser.add_argument('--mutant_eval_episodes', type=int, default=1000)

        # Dispatching
        parser.add_argument('--ppo_overrides', type=str, default='', help='These are arguments you want to pass to PPO')
        # parser.add_argument('--br_overrides', type=str, default='', help='These are arguments you want to pass to BR. DO NOT INCLUDE EVAL ARGS HERE')
        # parser.add_argument('--eval_overrides', type=str, default='', help="These are arguments you want to pass directly through to evaluate. They ALSO get passed to best respones")

    def handle(self, *args, **options):
        setup_logging()
        opts = AttrDict(options)

        if opts.game_name != 'python_clock_auction':
            raise ValueError("Only python_clock_auction is supported for now")

        metric = METRICS.get(opts.fitness_function)
        if metric is None:
            raise ValueError(f"Invalid fitness function. Known choices are {METRICS.keys()}")

        # STEP 1: Dispatch an initial population of PPOs. 
        # TODO: Is a different seed enough to make a difference, or should we mutate straight away so they don't have uniform starting strategies?
        dispatch_experiments(opts.network_config_file, base_job_name=opts.experiment_name, game_name=opts.filename, overrides=opts.ppo_overrides, n_seeds=opts.population_size)

        generation = -2
        while generation < opts.max_generations:
            generation += 2
            # STEP 2: Poll for PPOs having finished. They might as well evaluate themselves too.
            while True:
                time.sleep(opts.poll_interval)
                finished_qs = Evaluation.objects.filter(
                    checkpoint__equilibrium_solver_run__experiment__name=opts.experiment_name,
                    checkpoint__equilibrium_solver_run__generation=generation,
                )
                finished_count = finished_qs.count()
                logger.info("Finished %d/%d", finished_count, opts.population_size)
                if finished_count == opts.population_size:
                    break
                else:
                    logger.info("Going to sleep for %f seconds", opts.poll_interval)
            
            # STEP 3: SELECT PARENTS TO MUTATE
            fitnesses = []
            for parent in finished_qs:
                fitness = metric(parent.samples)
                fitnesses.append(fitness) 
            
            # STEP 4: MAKE MUTATIONS
            # TODO: temperature
            sample_probs = np.exp(fitnesses) / np.sum(np.exp(fitnesses))
            mutants = []
            mutant_fitnesses = []
            for i in range(opts.candidate_mutant_size):
                parent_index = np.random.choice(opts.population_size, p=sample_probs)
                parent = finished_qs[parent_index]
                parent_env_and_policy = ppo_db_checkpoint_loader(parent.checkpoint) # TODO: Could cache these
                mutant_env_and_policy = parent_to_mutation(parent_env_and_policy, num_states_to_save=opts.sm_states, mag=opts.mutation_strength)
                # STEP 5: EVAL MUTANTS (note: could be parallel)
                env = EnvParams(track_stats=True).make_env(mutant_env_and_policy.game)
                eval_dict = eval_agents_parallel(env, mutant_env_and_policy.agents, opts.mutant_eval_episodes)
                eval_dict.update(AuctionStatTrackingDecorator.merge_stats(env))
                mutant_fitness = metric(eval_dict)
                mutant_fitnesses.append(mutant_fitness)
                mutants.append({
                    'mutant': mutant_env_and_policy,
                    'parent': parent_index,
                    'walltime': eval_dict['walltime'],
                })

            # STEP 6: SELECT MUTANTS TO KEEP (MOST FIT)
            idx = np.argsort(mutant_fitnesses)[::-1][:opts.population_size]

            # STEP 7: LAUNCH MUTANTS
            for i in idx:
                mutant = mutants[i]
                mutant_run = EquilibriumSolverRun.objects.create(
                    experiment=Experiment.objects.get(name=opts.experiment_name),
                    generation=generation + 1,
                    name=f"MUTANT_{i}_FOR_GENERATION_{generation + 1}",
                    game=get_or_create_game(opts.game_name),
                    parent=finished_qs[mutant['parent']]
                )
                checkpoint = EquilibriumSolverRunCheckpoint.objects.create(
                    equilibrium_solver_run=mutant_run,
                    t=0,
                    walltime=mutant['walltime'],
                    policy=pickle.dumps(mutant['mutant'].make_policy().save())
                )

                mutant_job_name = mutant_run.name
                dispatch_from_checkpoint(checkpoint.pk, opts.game_name, opts.network_config_file, opts.experiment_name, mutant_job_name, overrides=opts.ppo_overrides)

        # STEP 2: Poll for PPOs having finished. They might as well evaluate themselves too.
        while True:
            time.sleep(opts.poll_interval)
            finished_qs = Evaluation.objects.filter(
                checkpoint__equilibrium_solver_run__experiment__name=opts.experiment_name,
                checkpoint__generation=generation + 2,
            )
            finished_count = finished_qs.count()
            logger.info("Finished %d/%d", finished_count, opts.population_size)
            if finished_count == opts.population_size:
                break
            else:
                logger.info("Going to sleep for %f seconds", opts.poll_interval)

        # NASHCONV? BRs?
        logger.info("HAVE A GOOD DAY")