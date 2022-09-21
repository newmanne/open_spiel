from django.core.management.base import BaseCommand
from open_spiel.python.examples.ubc_utils import apply_optional_overrides, fix_seeds, default_device, apply_optional_overrides, random_string
from open_spiel.python.examples.ppo_utils import read_ppo_config
import logging
from auctions.models import *
import sys
from auctions.webutils import *
from auctions.management.commands.ppo_eval import eval_command, EvalDefaults
from distutils import util
from auctions.savers import DBBRResultSaver
from open_spiel.python.examples.ppo_br import run_br

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs BR and saves the results'

    def add_arguments(self, parser):
        parser.add_argument('--total_timesteps', type=int, required=True)
        
        parser.add_argument('--br_player', type=int, default=0)
        parser.add_argument('--br_name', type=str)
        parser.add_argument('--config', type=str, required=True)
        
        parser.add_argument('--report_freq', type=int, default=100)
        parser.add_argument('--compute_exact_br', type=bool, default=False, help='Whether to compute an exact best response. Usually not possible')
        parser.add_argument('--dry_run', type=bool, default=False, help='If true, do not save')
        parser.add_argument('--device', type=str, default=default_device)
        parser.add_argument('--seed', type=int, default=1234)

        # WANDB
        parser.add_argument('--use_wandb', type=util.strtobool, default=1)
        parser.add_argument('--wandb_note', type=str, default='') 

        # Rewards dispatching
        parser.add_argument('--dispatch_rewards', type=util.strtobool, default=0)
        parser.add_argument('--eval_num_samples', type=int, default=EvalDefaults.DEFAULT_NUM_SAMPLES)
        parser.add_argument('--eval_report_freq', type=int, default=EvalDefaults.DEFAULT_REPORT_FREQ)
        parser.add_argument('--eval_compute_efficiency', type=util.strtobool, default=0)

        parser.add_argument('--t', type=int)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)

    def handle(self, *args, **options):
        setup_logging()
        opts = AttrDict(options)
        t = opts.t
        experiment_name = opts.experiment_name
        run_name = opts.run_name
        config_name = opts.config
        br_player = opts.br_player
        dry_run = opts.dry_run

        fix_seeds(opts.seed)

        # Find the equilibrium_solver_run_checkpoint that it's responding to
        equilibrium_solver_run_checkpoint = get_checkpoint_by_name(experiment_name, run_name, t)

        # Build the result saver
        result_saver = DBBRResultSaver(equilibrium_solver_run_checkpoint, config_name, dry_run)

        # Load the environment from the database
        env_and_policy = ppo_db_checkpoint_loader(equilibrium_solver_run_checkpoint)

        # Read config from file system and apply command line overrides
        config = read_ppo_config(config_name)
        apply_optional_overrides(opts, sys.argv, config)

        if opts.use_wandb:
            import wandb
            wandb.init(project=experiment_name, entity="ubc-algorithms", notes=opts.wandb_note, config=config, tags=[equilibrium_solver_run_checkpoint.equilibrium_solver_run.game.name, run_name], job_type='BR')

        # Run best response
        run_br(env_and_policy, br_player, opts.total_timesteps, config, report_freq=opts.report_freq, result_saver=result_saver, seed=opts.seed, compute_exact_br=opts.compute_exact_br)

        # Evaluation
        if opts.dispatch_rewards:
            eval_command(opts.t, opts.experiment_name, opts.run_name, config_name, opts.br_player, opts.dry_run, opts.seed, opts.eval_report_freq, opts.eval_num_samples, opts.eval_compute_efficiency)