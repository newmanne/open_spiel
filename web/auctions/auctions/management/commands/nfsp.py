from django.core.management.base import BaseCommand
from open_spiel.python.examples.ubc_nfsp_example import run_nfsp, setup_directory_structure
from open_spiel.python.examples.ubc_utils import smart_load_sequential_game, load_game_config, fix_seeds, read_config, apply_optional_overrides, add_optional_overrides, default_device
import sys
import logging
import pickle
from auctions.models import *
from auctions.webutils import *
import json
import open_spiel.python.examples.ubc_dispatch as dispatch
from distutils import util

logger = logging.getLogger(__name__)

class DBNFSPSaver:

    def __init__(self, eq_solver_run):
        self.equilibrium_solver_run = eq_solver_run

    def save(self, result):
        episode = result['episode']

        logger.info(f"Saving episode {episode} to DB")

        EquilibriumSolverRunCheckpoint.objects.create(
            equilibrium_solver_run = self.equilibrium_solver_run,
            walltime = result['walltime'],
            nash_conv = None, # Probably never used
            approx_nash_conv = None, # To be added later
            policy = pickle.dumps(result['policy']),
            t = episode,
        )
        
        return episode

class DBBRDispatcher:

    def __init__(self, num_players, eval_overrides, br_overrides, eq_solver_run, br_portfolio_path):
        self.num_players = num_players
        self.eval_overrides = eval_overrides
        self.br_overrides = br_overrides
        self.eq_solver_run = eq_solver_run
        self.br_portfolio_path = br_portfolio_path

    def dispatch(self, t):
        eq = self.eq_solver_run
        for player in range(self.num_players):
            dispatch.dispatch_br_database(eq.experiment.name, eq.name, t, player, self.br_portfolio_path, overrides=self.br_overrides + f' --eval_overrides "{self.eval_overrides}"')
            dispatch.dispatch_eval_database(eq.experiment.name, eq.name, t, player, 'straightforward', overrides=self.eval_overrides) # Straightforward eval
        dispatch.dispatch_eval_database(eq.experiment.name, eq.name, t, None, None, overrides=self.eval_overrides)
            

class Command(BaseCommand):
    help = 'Runs NFSP and saves the results'

    def add_arguments(self, parser):
        parser.add_argument('--num_training_episodes', type=int, required=True)
        parser.add_argument('--iterate_br', type=util.strtobool, default=1)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--network_config_file', type=str, default='network.yml')
        parser.add_argument('--compute_nash_conv', type=bool, default=False)
        parser.add_argument('--device', type=str, default=default_device)

        # Directory
        parser.add_argument('--output_dir', type=str, default='output') # Note: DONT NAME THIS "checkpoints" because of a jupyter notebook
        parser.add_argument('--warn_on_overwrite', type=bool, default=False)

        # Naming
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--job_name', type=str, default='auction')
        parser.add_argument('--filename', type=str, default='parameters.json') # Select clock auction game
        parser.add_argument('--game_name', type=str, default='clock_auction')
        
        # Reporting and evaluation
        parser.add_argument('--report_freq', type=int, default=50_000)
        parser.add_argument('--eval_every', type=int, default=300_000)
        parser.add_argument("--eval_every_early", type=int, default=None)
        parser.add_argument("--eval_exactly", nargs="+", default=[], type=int)
        parser.add_argument("--eval_zero", type=util.strtobool, default=1)

        # Dispatching
        parser.add_argument('--dispatch_br', type=util.strtobool, default=1)
        parser.add_argument('--br_portfolio_path', type=str, default=None)
        parser.add_argument('--br_overrides', type=str, default='')
        parser.add_argument('--eval_overrides', type=str, default='')

        add_optional_overrides(parser)

    def handle(self, *args, **options):
        setup_logging()

        opts = AttrDict(options)
        game_name = opts.filename
        run_name = opts.job_name
        config_name = opts.network_config_file
        experiment_name = opts.experiment_name
        seed = opts.seed

        fix_seeds(seed)

        # 0) Read the config file
        config = read_config(config_name)
        apply_optional_overrides(opts, sys.argv, config)
        
        logging.info(f'Network params: {config}')
        logging.info(f'Command line commands {opts}')

        # 1) Make the game if it doesn't exist
        try:
            game = Game.objects.get(name=game_name)
            game_config = game.config
        except Game.DoesNotExist:
            game_obj = smart_load_sequential_game('clock_auction', dict(filename=game_name))
            game_config = load_game_config(game_name)
            game, _ = Game.objects.get_or_create( 
                name=game_name,
                num_players=game_obj.num_players(),
                num_actions=game_obj.num_distinct_actions(),
                num_products=len(game_config['licenses']),
                config=game_config
            )

        # 2) Make the experiment if it doesn't exist
        experiment, _ = Experiment.objects.get_or_create(name=experiment_name)

        output_dir = f'{OUTPUT_ROOT}/{experiment_name}/{run_name}'
        setup_directory_structure(output_dir, opts.warn_on_overwrite)

        # Save the game config so there's no confusion later if you need to cross-reference. Shouldn't techincally need this in the database version, but why not
        with open(f'{output_dir}/game.json', 'w') as outfile:
            json.dump(game.config, outfile)

        # 3) Make an EquilibriumSolverRun
        eq_solver_run = EquilibriumSolverRun.objects.create(
            experiment=experiment,
            name=run_name,
            game=game,
            config=config
        )

        # Load the environment from the database
        env_and_model = env_and_model_from_run(eq_solver_run)

        result_saver = DBNFSPSaver(eq_solver_run=eq_solver_run)
        dispatcher = DBBRDispatcher(game.num_players, opts.eval_overrides, opts.br_overrides, eq_solver_run, opts.br_portfolio_path)

        run_nfsp(env_and_model, opts.num_training_episodes, opts.iterate_br, result_saver, seed, opts.compute_nash_conv, dispatcher, opts.eval_every, opts.eval_every_early, opts.eval_exactly, opts.eval_zero, opts.report_freq, opts.dispatch_br)