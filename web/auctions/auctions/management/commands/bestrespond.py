from django.core.management.base import BaseCommand
from open_spiel.python.examples.ubc_br import run_br, add_argparse_args
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, load_game_config, read_config, apply_optional_overrides, fix_seeds
from open_spiel.python.examples.ubc_nfsp_example import setup
from open_spiel.python.examples.ubc_dispatch import dispatch_eval_database
import logging
import pickle
from auctions.models import *
import os
import sys
from auctions.webutils import *
import yaml
import open_spiel.python.examples.ubc_dispatch as dispatch

logger = logging.getLogger(__name__)

class DBBRResultSaver:

    def __init__(self, equilibrium_solver_run_checkpoint, br_name, dry_run):
        self.equilibrium_solver_run_checkpoint = equilibrium_solver_run_checkpoint
        self.br_name = br_name
        self.dry_run = dry_run

    def save(self, checkpoint):
        if not self.dry_run:
            BestResponse.objects.create(
                checkpoint = self.equilibrium_solver_run_checkpoint,
                br_player = checkpoint['br_player'],
                walltime = checkpoint['walltime'],
                model = pickle.dumps(checkpoint['agent']),
                config = checkpoint['config'],
                name = self.br_name,
                t = checkpoint['episode']
            )

class Command(BaseCommand):
    help = 'Runs BR and saves the results'

    def add_arguments(self, parser):
        add_argparse_args(parser)
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
        equilibrium_solver_run_checkpoint = get_checkpoint(experiment_name, run_name, t)

        # Build the result saver
        result_saver = DBBRResultSaver(equilibrium_solver_run_checkpoint, config_name, dry_run)

        # Load the environment from the database
        env_and_model = db_checkpoint_loader(equilibrium_solver_run_checkpoint)

        # Read config from file system and apply command line overrides
        config = read_config(config_name)

        apply_optional_overrides(opts, sys.argv, config)

        # Run best response
        run_br(result_saver, opts.report_freq, env_and_model, opts.num_training_episodes, br_player, opts.dry_run, opts.seed, opts.compute_exact_br, config)

        if opts.dispatch_rewards and not dry_run:
            dispatch.dispatch_eval_database(experiment_name, run_name, t, br_player, config_name, overrides=opts.eval_overrides)