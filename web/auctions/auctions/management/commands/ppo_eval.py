from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_eval import run_eval, EvalDefaults
from open_spiel.python.examples.ubc_utils import series_to_quantiles, fix_seeds
import logging
from auctions.models import *
from auctions.webutils import *
from open_spiel.python.examples.ubc_decorators import TakeSingleActionDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from distutils import util

logger = logging.getLogger(__name__)

def eval_command(t, experiment_name, run_name, br_name, br_player, dry_run, seed, report_freq, num_samples, compute_efficiency, num_envs=EvalDefaults.DEFAULT_NUM_ENVS):
    fix_seeds(seed)

    logging.info("EVALUATION STARTING")
    # Find the equilibrium_solver_run_checkpoint 
    equilibrium_solver_run_checkpoint = get_checkpoint_by_name(experiment_name, run_name, t)

    # Load the environment
    env_params = EnvParams(track_stats=True, seed=seed, num_envs=num_envs)
    env_and_policy = ppo_db_checkpoint_loader(equilibrium_solver_run_checkpoint, env_params=env_params)

    # Replace agents if necessary
    if br_name is None:
        logging.info("No best reponders provided. Just evaluating the policy")
    elif br_name == 'straightforward':
        logging.info(f"Agent {br_player} will be straightforward")
        game = env_and_policy.game
        env_and_policy.agents[br_player] = TakeSingleActionDecorator(StraightforwardAgent(br_player, game), game.num_distinct_actions())

        if not dry_run:
            # Create a "fake" listing for the straightforward BR
            best_response = BestResponse.objects.create(
                checkpoint = equilibrium_solver_run_checkpoint,
                br_player = br_player,
                walltime = 0.,
                model = None,
                config = dict(),
                name = 'straightforward',
                t = 0,
            )
    else:
        logging.info(f"Reading from agent {br_name}")
        best_response = BestResponse.objects.get(
            checkpoint = equilibrium_solver_run_checkpoint,
            br_player = br_player,
            name = br_name
        )
        br_agent = load_ppo_agent(best_response)
        env_and_policy.agents[br_player] = br_agent 

    # RUN EVAL
    eval_output = run_eval(env_and_policy, num_samples=num_samples, report_freq=report_freq, seed=seed, compute_efficiency=compute_efficiency)

    # SAVE EVAL
    if not dry_run:
        if br_name is None:
            # Save a full evaluation with all of the samples
            mean_rewards = []
            for player in range(equilibrium_solver_run_checkpoint.equilibrium_solver_run.game.num_players):
                mean_rewards.append(
                    pd.Series(eval_output['raw_rewards'][player]).mean()
                )
            
            eval_output = convert_pesky_np(eval_output)
            Evaluation.objects.create(
                walltime = eval_output.pop('walltime'),
                checkpoint = equilibrium_solver_run_checkpoint,
                samples = eval_output,
                mean_rewards = mean_rewards
            )
            
        else:
            br_rewards = pd.Series(eval_output['raw_rewards'][br_player])
            BREvaluation.objects.create(
                best_response = best_response,
                walltime = eval_output['walltime'],
                expected_value_cdf = series_to_quantiles(br_rewards),
                expected_value_stats = br_rewards.describe().to_dict()
            )
        logging.info("Saved to DB")
    # TODO: Using an atomic transaction, make changes to the approx_nash_conv field?



class Command(BaseCommand):
    help = 'Evaluates a policy and saves the result'

    def add_arguments(self, parser):
        parser.add_argument('--eval_num_samples', type=int, default=EvalDefaults.DEFAULT_NUM_SAMPLES)
        parser.add_argument('--eval_report_freq', type=int, default=EvalDefaults.DEFAULT_REPORT_FREQ)
        parser.add_argument('--eval_num_envs', type=int, default=EvalDefaults.DEFAULT_NUM_ENVS)
        parser.add_argument('--eval_compute_efficiency', type=util.strtobool, default=EvalDefaults.DEFAULT_COMPUTE_EFFICIENCY)
        parser.add_argument('--seed', type=int, default=EvalDefaults.DEFAULT_SEED)
        parser.add_argument('--br_name', type=str, default=None)

        # Needed to identify the checkpoint
        parser.add_argument('--t', type=int)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)

        # Needed to identify the BR along with br_name
        parser.add_argument('--br_player', type=int)
        parser.add_argument('--dry_run', type=util.strtobool, default=0)

        add_profiling_flags(parser)

    def handle(self, *args, **options):
        setup_logging()
        opts = AttrDict(options)

        cmd = lambda: eval_command(opts.t, opts.experiment_name, opts.run_name, opts.br_name, opts.br_player, opts.dry_run, opts.seed, opts.eval_report_freq, opts.eval_num_samples, opts.eval_compute_efficiency, num_envs=opts.eval_num_envs)
        profile_cmd(cmd, opts.pprofile, opts.pprofile_file, opts.cprofile, opts.cprofile_file)
