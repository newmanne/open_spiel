from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_utils import EpisodeTimer
from open_spiel.python.examples.cfr_utils import read_cfr_config, load_solver, make_cfr_agent
from open_spiel.python.examples.ubc_utils import fix_seeds, apply_optional_overrides, setup_directory_structure, time_bounded_run
from open_spiel.python.examples.ubc_cma import get_game_info
from open_spiel.python.examples.ubc_decorators import ModalAgentDecorator
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.rl_agent_policy import JointRLAgentPolicy
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
import wandb


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
        dispatcher = DBBRDispatcher(game_db.num_players, opts.eval_overrides, opts.br_overrides, eq_solver_run, opts.br_portfolio_path, opts.dispatch_br, opts.eval_inline, opts.profile_memory, opts.use_wandb) if not opts.dry_run else None
        eval_episode_timer = EpisodeTimer(opts.eval_every, early_frequency=opts.eval_every_early, fixed_episodes=opts.eval_exactly, eval_zero=opts.eval_zero, every_seconds=opts.eval_every_seconds)
        report_timer = EpisodeTimer(opts.report_freq)

        game = game_db.load_as_spiel()
        solver = load_solver(config, game)

        game_info = get_game_info(game, game_db)

        if opts.use_wandb:
            import wandb
            config['track_stats'] = True
            config['clear_on_report'] = True
            config['game_name'] = game_name
            config['cfg'] = config_name
            config.update(game_info)
            wandb.init(project=experiment_name, entity="ubc-algorithms", name=run_name, notes=opts.wandb_note, config=config)
            # configure wandb to use "global_step" as the x axis for all plots
            wandb.define_metric("*", step_metric="global_step")


        cmd = lambda: run_cfr(config, game, solver, opts.total_timesteps, result_saver=result_saver, seed=seed, compute_nash_conv_on_report=opts.compute_nash_conv, dispatcher=dispatcher, report_timer=report_timer, eval_timer=eval_episode_timer, use_wandb=opts.use_wandb, time_limit_seconds=opts.time_limit_seconds)
        profile_cmd(cmd, opts.pprofile, opts.pprofile_file, opts.cprofile, opts.cprofile_file)

        logging.info("All done. Goodbye")


def trigger(solver, i, start_time, result_saver, dispatcher, use_wandb=False):
    policy = solver.average_policy()
    checkpoint = dict(episode=i, walltime=time.time() - start_time, policy=policy)
    if result_saver is not None:
        checkpoint_name = result_saver.save(checkpoint)
        if dispatcher is not None:
            dispatcher.dispatch(checkpoint_name, policy, solver._game)
            # if use_wandb:
            #     wandb.log({}, step=i, commit=True)


def compute_nash_conv(game, solver, time_limit_seconds, restrict_to_heuristics=False):
    avg_policy = solver.average_policy()
    agents = {}
    for player_id in range(game.num_players()):
        agents[player_id] = make_cfr_agent(player_id, None, None)
        agents[player_id].policy = avg_policy
        agents[player_id] = ModalAgentDecorator(agents[player_id])
    policy = JointRLAgentPolicy(game, agents, False)

    worked, nash_conv_runtime, res = time_bounded_run(time_limit_seconds, nash_conv, game, policy, return_only_nash_conv=False, restrict_to_heuristics=restrict_to_heuristics)
    if worked:
        (nc, heuristic_conv_player_improvements, br_policies) = res
    else:
        (nc, heuristic_conv_player_improvements, br_policies) = (None, None, None)
    return worked, nc, heuristic_conv_player_improvements, nash_conv_runtime

def run_cfr(solver_config, game, solver, total_timesteps, result_saver=None, seed=1234, dispatcher=None, report_timer=None, eval_timer=None, use_wandb=False, compute_nash_conv_on_report=False, time_limit_seconds=None):
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
            wandb_data = {}

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
                    wandb_data[f'{wrapper.__qualname__}_hit_rate'] = hit_rate

            # solver stats
            solver_stats = solver.get_solver_stats()
            wandb_data.update({f'mccfr_{k}': v for k, v in solver_stats.items()})

            # NashConv (TODO: add flag for heuristicconv)
            if compute_nash_conv_on_report:
                logger.info("Computing NashConv...")
                nc_worked, nc, nash_conv_player_improvements, nash_conv_runtime = compute_nash_conv(game, solver, 300)
                if nc_worked: 
                    logger.info(f"NashConv: {nc:.3f} (computed in {nash_conv_runtime:.3f} seconds)")
                    for player in range(len(nash_conv_player_improvements)):
                        logger.info(f"Player {player} improvement: {nash_conv_player_improvements[player]}")

                    wandb_data.update({
                        'nash_conv': nc, 
                        'nash_conv_runtime': nash_conv_runtime,
                        **{f'nash_conv_player_improvements_{p}': v for p, v in enumerate(nash_conv_player_improvements)},
                    })
                else:
                    logger.info("NashConv timed out")
            else:
                logger.info('Skipping NashConv.')

            # TODO: These caches I would love to see, as they are the most important. But they aren't instrumented because they aren't from a decorator...
            # logger.info(f"State cache stats: {game.state_cache.cache_info()}")
            # logger.info(f"Lottery cache stats: {game.lottery_cache.cache_info()}")

            if use_wandb:
                wandb.log({**wandb_data, 'global_step': i})

        if solver_config['solver'] == 'mccfr':
            solver.iteration()
        else:
            solver.evaluate_and_update_policy()

        if eval_timer and (eval_timer.should_trigger(i)):
            trigger(solver, i, start_time, result_saver, dispatcher, use_wandb=use_wandb)

    # One last time
    trigger(solver, i, start_time, result_saver, dispatcher, use_wandb=use_wandb)