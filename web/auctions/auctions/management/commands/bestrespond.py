from django.core.management.base import BaseCommand
from open_spiel.python.examples.ubc_br import run_br
from open_spiel.python.examples.ubc_utils import read_config, apply_optional_overrides, fix_seeds, add_optional_overrides, default_device
import logging
import pickle
from auctions.models import *
import sys
from auctions.webutils import *
from auctions.management.commands.evaluate import eval_command
import open_spiel.python.examples.ubc_dispatch as dispatch
from distutils import util
from open_spiel.python.examples.ubc_evaluate_policy import DEFAULT_NUM_SAMPLES, run_eval, DEFAULT_REPORT_FREQ

logger = logging.getLogger(__name__)

class DBBRResultSaver:

    def __init__(self, equilibrium_solver_run_checkpoint, br_name, dry_run):
        self.equilibrium_solver_run_checkpoint = equilibrium_solver_run_checkpoint
        self.br_name = br_name
        self.dry_run = dry_run
        self.result = None

    def save(self, checkpoint):
        if not self.dry_run:
            self.result = BestResponse.objects.create(
                checkpoint = self.equilibrium_solver_run_checkpoint,
                br_player = checkpoint['br_player'],
                walltime = checkpoint['walltime'],
                model = pickle.dumps(checkpoint['agent']),
                config = checkpoint['config'],
                name = self.br_name,
                t = checkpoint['episode']
            )

    def get_result(self):
        return self.result

class Command(BaseCommand):
    help = 'Runs BR and saves the results'

    def add_arguments(self, parser):
        parser.add_argument('--num_training_episodes', type=int, required=True)
        
        parser.add_argument('--br_player', type=int, default=0)
        parser.add_argument('--br_name', type=str)
        parser.add_argument('--config', type=str, required=True)
        
        parser.add_argument('--report_freq', type=int, default=50_000)
        parser.add_argument('--compute_exact_br', type=bool, default=False, help='Whether to compute an exact best response. Usually not possible')
        parser.add_argument('--dry_run', type=bool, default=False, help='If true, do not save')
        parser.add_argument('--device', type=str, default=default_device)
        parser.add_argument('--seed', type=int, default=1234)

        # Rewards dispatching
        parser.add_argument('--dispatch_rewards', type=util.strtobool, default=0)
        parser.add_argument('--eval_num_samples', type=int, default=DEFAULT_NUM_SAMPLES)
        parser.add_argument('--eval_report_freq', type=int, default=DEFAULT_REPORT_FREQ)

        parser.add_argument('--t', type=int)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)
        add_optional_overrides(parser)

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
        env_and_model = db_checkpoint_loader(equilibrium_solver_run_checkpoint)

        # Read config from file system and apply command line overrides
        config = read_config(config_name)
        apply_optional_overrides(opts, sys.argv, config)

        # Run best response
        run_br(result_saver, opts.report_freq, env_and_model, opts.num_training_episodes, br_player, opts.dry_run, opts.seed, opts.compute_exact_br, config)

        # Evaluation
        if opts.dispatch_rewards:
            eval_command(opts.t, opts.experiment_name, opts.run_name, config_name, opts.br_player, opts.dry_run, opts.seed, opts.eval_report_freq, opts.eval_num_samples)