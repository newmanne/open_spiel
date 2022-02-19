from django.core.management.base import BaseCommand
from open_spiel.python.examples.ubc_evaluate_policy import run_eval, add_argparse_args
from open_spiel.python.examples.ubc_utils import series_to_quantiles, fix_seeds
import logging
import pickle
from auctions.models import *
import os
from auctions.webutils import *

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Evaluates a policy and saves the result'

    def add_arguments(self, parser):
        add_argparse_args(parser)

        # Needed to identify the checkpoint
        parser.add_argument('--t', type=int)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)

        # Needed to identify the BR along with br_name
        parser.add_argument('--br_player', type=int)

    def handle(self, *args, **options):
        setup_logging()
        opts = AttrDict(options)

        fix_seeds(opts.seed)

        t = opts.t
        experiment_name = opts.experiment_name
        run_name = opts.run_name
        br_name = opts.br_name
        br_player = opts.br_player

        # Find the equilibrium_solver_run_checkpoint 
        equilibrium_solver_run_checkpoint = EquilibriumSolverRunCheckpoint.objects.get(
            t=t,
            equilibrium_solver_run__name=run_name,
            equilibrium_solver_run__experiment__name=experiment_name
        )

        # Load the environment
        env_and_model = db_checkpoint_loader(equilibrium_solver_run_checkpoint)

        # Replace agents if necessary
        if br_name is None:
            logging.info("No best reponders provided. Just evaluating the policy")
        elif br_name == 'straightforward':
            logging.info(f"Agent {br_player} will be straightforward")
            game, game_config = env_and_model.game, env_and_model.game_config
            env_and_model.agents[br_player] = TakeSingleActionDecorator(StraightforwardAgent(br_player, game_config, game.num_distinct_actions()), game.num_distinct_actions())
        else:
            logging.info(f"Reading from agent {br_name}")
            best_response = BestResponse.get(
                checkpoint = equilibrium_solver_run_checkpoint,
                br_player = br_player,
                name = br_name
            )
            br_agent = load_dqn_agent(best_respone)
            env_and_model.agents[br_player] = br_agent 

        # RUN EVAL
        eval_output = run_eval(env_and_model, opts.num_samples, opts.report_freq, opts.seed)

        if br_name is None:
            del eval_output['br_name']
            del eval_output['br_agent']

            # Save a full evaluation with all of the samples
            mean_rewards = []
            for player in range(equilibrium_solver_run_checkpoint.equilibrium_solver_run.game.num_players):
                mean_rewards.append(
                    pd.Series(eval_output['rewards'][player]).mean()
                )
            Evaluation.objects.create(
                walltime = eval_output.pop('walltime'),
                checkpoint = equilibrium_solver_run_checkpoint,
                samples = eval_output,
                mean_rewards = mean_rewards
            )
        else:
            BREvaluation.objects.create(
                best_response = best_response,
                walltime = eval_output['walltime'],
                expected_value_cdf = series_to_quantiles(eval_output['rewards'][br_player])
                expected_value_stats = eval_output['rewards'][br_player].describe()
            )

        # TODO: Using an atomic transaction, make changes to the approx_nash_conv field?