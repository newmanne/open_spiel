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

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs PPO and saves the results'

    def add_arguments(self, parser):
        parser.add_argument('--total_timesteps', type=int, required=True)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--network_config_file', type=str, default='network.yml')
        parser.add_argument('--compute_nash_conv', type=bool, default=False)
        parser.add_argument('--device', type=str, default=default_device)

        parser.add_argument('--overwrite_db', type=util.strtobool, default=0)
        parser.add_argument('--dry_run', type=util.strtobool, default=0)

        # Directory
        parser.add_argument('--output_dir', type=str, default='output') # Note: DONT NAME THIS "checkpoints" because of a jupyter notebook bug
        parser.add_argument('--warn_on_overwrite', type=bool, default=False)

        # Naming
        add_experiment_flags(parser)

        # Start From Checkpoint
        parser.add_argument('--parent_checkpoint_pk', type=int, default=None)

        # Potential
        parser.add_argument('--potential_function', type=str, default=None) 
        parser.add_argument('--scale_coef', type=float, default=None)

        # Reporting and evaluation
        add_reporting_flags(parser)

        # WANDB
        add_wandb_flags(parser)

        # Dispatching
        add_dispatching_flags(parser)

        # Profiling
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
        config = read_ppo_config(config_name)
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

        parent_checkpoint = None
        if opts.parent_checkpoint_pk is not None:
            parent_checkpoint = EquilibriumSolverRunCheckpoint.objects.get(pk=opts.parent_checkpoint_pk)
        
        # Parse env params
        env_params = EnvParams.from_config(config)

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
            generation = 0 if parent_checkpoint is None else parent_checkpoint.equilibrium_solver_run.generation + 1
            eq_solver_run = EquilibriumSolverRun.objects.create(experiment=experiment, name=run_name, game=game_db, config=config, parent=parent_checkpoint, generation=generation)

            # Load the environment from the database
            if opts.parent_checkpoint_pk is not None:
                env_and_policy = ppo_db_checkpoint_loader(parent_checkpoint, env_params=env_params)
            else:
                env_and_policy = env_and_policy_from_run(eq_solver_run, env_params=env_params)
        else:
            if opts.parent_checkpoint_pk is not None:
                env_and_policy = ppo_db_checkpoint_loader(parent_checkpoint, env_params=env_params)
            else:
                env_and_policy = env_and_policy_for_dry_run(game_db, config, env_params=env_params)

        result_saver = DBPolicySaver(eq_solver_run=eq_solver_run) if not opts.dry_run else None
        dispatcher = DBBRDispatcher(game_db.num_players, opts.eval_overrides, opts.br_overrides, eq_solver_run, opts.br_portfolio_path, opts.dispatch_br, opts.eval_inline) if not opts.dry_run else None
        eval_episode_timer = EpisodeTimer(opts.eval_every, early_frequency=opts.eval_every_early, fixed_episodes=opts.eval_exactly, eval_zero=opts.eval_zero)
        report_timer = EpisodeTimer(opts.report_freq)

        cmd = lambda: run_ppo(env_and_policy, opts.total_timesteps, result_saver=result_saver, seed=seed, compute_nash_conv=opts.compute_nash_conv, dispatcher=dispatcher, report_timer=report_timer, eval_timer=eval_episode_timer, use_wandb=opts.use_wandb, wandb_step_interval=opts.wandb_step_interval)
        profile_cmd(cmd, opts.pprofile, opts.pprofile_file, opts.cprofile, opts.cprofile_file)
        