from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_utils import EpisodeTimer
from open_spiel.python.examples.cfr_utils import read_cfr_config, load_solver, make_cfr_agent
from open_spiel.python.examples.ubc_utils import fix_seeds, apply_optional_overrides, setup_directory_structure, time_bounded_run
from open_spiel.python.examples.ubc_cma import get_game_info, get_modal_nash_conv_new_rho
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
    help = 'Compute NC for a given checkpoint'

    def add_arguments(self, parser):
        parser.add_argument('--time_limit_seconds', type=int, default=None)
        parser.add_argument('--heuristic', type=util.strtobool, default=0)

        parser.add_argument('--t', type=int, default=None)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)

    def handle(self, *args, **options):
        setup_logging()

        opts = AttrDict(options)
        equilibrium_solver_run_checkpoint = get_checkpoint_by_name(opts.experiment_name, opts.run_name, opts.t)
        env_and_model = ppo_db_checkpoint_loader(equilibrium_solver_run_checkpoint)
        game = env_and_model.game
        c = equilibrium_solver_run_checkpoint.equilibrium_solver_run.config

        # Get modal eval
        eval = equilibrium_solver_run_checkpoint.get_modal_eval()
        logging.info(f"""Status quo: NC {eval.nash_conv}
                        NC runtime: {eval.nash_conv_runtime}
                        NCPI: {eval.nash_conv_player_improvements}
                        HC: {eval.heuristic_conv}
                        HC Runtime: {eval.heuristic_conv_runtime}
                        HCPI: {eval.heuristic_conv_player_improvements}
        """)

        if (opts.heuristic and eval.heuristic_conv is not None) or (not opts.heuristic and eval.nash_conv is not None):
            logging.info("Why are you recomputing what I already know? Stopping")
            return
        else:
            logging.info("I do not conclusively know about NC")

        computation_type = 'HC' if opts.heuristic else 'NC'
        logging.info(f"Starting {computation_type} Computation")
        time_taken, retval = get_modal_nash_conv_new_rho(game, env_and_model.make_policy(), c, rho=0, return_only_nash_conv=False, restrict_to_heuristics=opts.heuristic, time_limit_seconds=opts.time_limit_seconds)
        logging.info(
            f"Finished after {time_taken}. {retval}"
        )

        (nc, nash_conv_player_improvements, br_policies) = retval

        if nc is not None:
            logging.info("Updating result")
            if opts.heuristic:
                eval.heuristic_conv = nc
                eval.heuristic_conv_runtime = time_taken
                eval.heuristic_conv_player_improvements = list(nash_conv_player_improvements)
            else:
                eval.nash_conv = nc
                eval.nash_conv_runtime = time_taken
                eval.nash_conv_player_improvements = list(nash_conv_player_improvements)
        

            eval.save()
        

        logging.info("Good bye!")

