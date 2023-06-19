from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_utils import EpisodeTimer
from open_spiel.python.examples.cfr_utils import read_cfr_config, load_solver
from open_spiel.python.examples.ubc_utils import fix_seeds, apply_optional_overrides, setup_directory_structure
import sys
import logging
from auctions.models import *
from auctions.webutils import *
from auctions.savers import DBBRDispatcher, DBPolicySaver
import json
from distutils import util
from pathlib import Path
import time
import humanize
import functools
import gc

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs (MC)CFR and saves the results'

    def add_arguments(self, parser):
        parser.add_argument('--total_timesteps', type=int, required=True)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--network_config_file', type=str, default='network.yml')
        parser.add_argument('--compute_nash_conv', type=bool, default=False)
        parser.add_argument('--time_limit_seconds', type=int, default=None)

        parser.add_argument('--overwrite_db', type=util.strtobool, default=0)
        parser.add_argument('--dry_run', type=util.strtobool, default=0)

        # Directory 
        parser.add_argument('--output_dir', type=str, default='output') # Note: DONT NAME THIS "checkpoints" because of a jupyter notebook bug
        parser.add_argument('--warn_on_overwrite', type=bool, default=False)

        add_experiment_flags(parser)
        add_reporting_flags(parser)
        add_wandb_flags(parser, default=False) # Could be useful, but you'd have to actually track things
        add_dispatching_flags(parser)
        add_profiling_flags(parser)

    def handle(self, *args, **options):
        setup_logging()

        opts = AttrDict(options)
        game_name = opts.filename
        run_name = opts.job_name
        config_name = opts.network_config_file
        experiment_name = opts.experiment_name
        seed = opts.seed
        overwrite_db = opts.overwrite_db
        
        fix_seeds(seed)

        # 0) Read the config file
        # TODO: Still a healthy amount of duplicated code....
        config = read_cfr_config(config_name) 
        apply_optional_overrides(opts, sys.argv, config)
        
        logging.info(f'Network params: {config}')
        logging.info(f'Command line commands {opts}')

        if opts.use_wandb:
            import wandb
            config['track_stats'] = True
            config['clear_on_report'] = True
            config['game_name'] = game_name
            config['cfg'] = config_name
            wandb.init(project=experiment_name, entity="ubc-algorithms", name=run_name, notes=opts.wandb_note, config=config, tags=[game_name])

        # 1) Make the game if it doesn't exist
        game_db = get_or_create_game(game_name)

        if not opts.dry_run:
            # 2) Make the experiment if it doesn't exist
            experiment, _ = Experiment.objects.get_or_create(name=experiment_name)

            output_dir = f'{OUTPUT_ROOT}/{experiment_name}/{run_name}'
            setup_directory_structure(output_dir, opts.warn_on_overwrite)

            # Save the game config so there's no confusion later if you need to cross-reference. Shouldn't techincally need this in the database version, but why not
            with open(f'{output_dir}/game.json', 'w') as outfile:
                json.dump(game_db.config, outfile)

            if overwrite_db:
                try:
                    EquilibriumSolverRun.objects.get(experiment=experiment, name=run_name, game=game_db).delete()
                except EquilibriumSolverRun.DoesNotExist:
                    pass

            # 3) Make an EquilibriumSolverRun
            eq_solver_run = EquilibriumSolverRun.objects.create(experiment=experiment, name=run_name, game=game_db, config=config, parent=None, generation=0)


        result_saver = DBPolicySaver(eq_solver_run=eq_solver_run) if not opts.dry_run else None
        dispatcher = DBBRDispatcher(game_db.num_players, opts.eval_overrides, opts.br_overrides, eq_solver_run, opts.br_portfolio_path, opts.dispatch_br, opts.eval_inline) if not opts.dry_run else None
        eval_episode_timer = EpisodeTimer(opts.eval_every, early_frequency=opts.eval_every_early, fixed_episodes=opts.eval_exactly, eval_zero=opts.eval_zero)
        report_timer = EpisodeTimer(opts.report_freq)

        game = game_db.load_as_spiel()
        solver = load_solver(config, game)

        cmd = lambda: run_cfr(config, game, solver, opts.total_timesteps, result_saver=result_saver, seed=seed, compute_nash_conv=opts.compute_nash_conv, dispatcher=dispatcher, report_timer=report_timer, eval_timer=eval_episode_timer, use_wandb=opts.use_wandb, time_limit_seconds=opts.time_limit_seconds)
        profile_cmd(cmd, opts.pprofile, opts.pprofile_file, opts.cprofile, opts.cprofile_file)

        logging.info("All done. Goodbye")


def trigger(solver, i, start_time, result_saver, dispatcher):
    print("--------")
    print(i)
    import os, psutil
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) # MiB
    policy = solver.average_policy()
    checkpoint = dict(episode=i, walltime=time.time() - start_time, policy=policy)
    if result_saver is not None:
        checkpoint_name = result_saver.save(checkpoint)
        if dispatcher is not None:
            dispatcher.dispatch(checkpoint_name)
    print("POST")
    import os, psutil
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) # MiB
    print("--------")



def run_cfr(solver_config, game, solver, total_timesteps, result_saver=None, seed=1234, dispatcher=None, report_timer=None, eval_timer=None, use_wandb=False, compute_nash_conv=False, time_limit_seconds=None):
    start_time = time.time()
    # TODO: Wandb see ppo_utils: How often should we do this? On report? Eval? 
    # RUN SOLVER

    if time_limit_seconds:
        logger.info(f"Running for a time limit of {time_limit_seconds} seconds")

    for i in range(total_timesteps):

        if time_limit_seconds and time.time() - start_time > time_limit_seconds:
            logger.info("Out of time!")
            break
        
        last_iter = i == total_timesteps - 1

        if report_timer is not None and report_timer.should_trigger(i) and i > 0:
            avg_iter_time = (time.time() - start_time) / i
            extrapolated_time = (total_timesteps - i) * avg_iter_time
            logger.info(f"Starting iteration {i}. At the current rate, you will finish in {humanize.naturaldelta(extrapolated_time, minimum_unit='seconds')}")
            logger.info(f"I average {1/avg_iter_time:.3f} iterations per second")

            ### CACHING STATS ###
            gc.collect()
            wrappers = [a for a in gc.get_objects() if isinstance(a, functools._lru_cache_wrapper)]

            for wrapper in wrappers:
                if 'ClockAuction' in wrapper.__qualname__:
                    cache_info = wrapper.cache_info()
                    if cache_info.misses == 0:
                        hit_rate = 0
                    else:
                        hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses)
                    logger.info(f"{wrapper.__qualname__}: Hit rate {hit_rate:.2%}")

            # TODO: These caches I would love to see, as they are the most important. But they aren't instrumented because they aren't from a decorator...
            # logger.info(f"State cache stats: {game.state_cache.cache_info()}")
            # logger.info(f"Lottery cache stats: {game.lottery_cache.cache_info()}")


        if solver_config['solver'] == 'mccfr':
            solver.iteration()
        else:
            solver.evaluate_and_update_policy()

        if eval_timer and (eval_timer.should_trigger(i)):
            trigger(solver, i, start_time, result_saver, dispatcher)

    # One last time
    trigger(solver, i, start_time, result_saver, dispatcher)