from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_eval import run_eval, EvalDefaults
from open_spiel.python.examples.ubc_utils import series_to_quantiles, fix_seeds, players_not_me, time_bounded_run
import logging
from auctions.models import *
from auctions.webutils import *
from open_spiel.python.examples.ubc_decorators import TakeSingleActionDecorator, TremblingAgentDecorator, ModalAgentDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from distutils import util
from open_spiel.python.algorithms.exploitability import nash_conv

logger = logging.getLogger(__name__)

def make_fake_br(equilibrium_solver_run_checkpoint, br_player, name):
    # Create a "fake" listing for the straightforward BR
    best_response = BestResponse.objects.create(
                checkpoint = equilibrium_solver_run_checkpoint,
                br_player = br_player,
                walltime = 0.,
                model = None,
                config = dict(),
                name = name,
                t = 0,
            )
    return best_response


def eval_command(t, experiment_name, run_name, br_mapping=None, dry_run=False, seed=EvalDefaults.DEFAULT_SEED, report_freq=EvalDefaults.DEFAULT_REPORT_FREQ, num_samples=EvalDefaults.DEFAULT_NUM_SAMPLES, compute_efficiency=False, num_envs=EvalDefaults.DEFAULT_NUM_ENVS, reseed=True, store_samples=True):
    import os, psutil
    logging.info("BEFORE EVAL")
    logging.info(str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) # MiB

    
    if br_mapping is None:
        br_mapping = dict()
    '''br_mapping is a dict with keys of player_ids mapping to br_names'''
    if reseed: # When you call this inline from somewhere else, you might not want to reseed quite so globally. Would be best if this function had a global RNG_STATE type thing to pass around...
        fix_seeds(seed)

    logging.info("EVALUATION STARTING")
    # Find the equilibrium_solver_run_checkpoint 
    equilibrium_solver_run_checkpoint = get_checkpoint_by_name(experiment_name, run_name, t)

    c = equilibrium_solver_run_checkpoint.equilibrium_solver_run.config
    cfr = c.get('solver_type') == 'cfr'
    env_params_kwargs = {'observer_params': dict(normalize=False)} if cfr else {}

    # Load the environment
    env_params = EnvParams(track_stats=True, seed=seed, num_envs=num_envs, **env_params_kwargs)
    env_and_policy = ppo_db_checkpoint_loader(equilibrium_solver_run_checkpoint, env_params=env_params)
    game = env_and_policy.game
    real_br = False
    name = ''

    # Replace agents if necessary
    if len(br_mapping) == 0:
        logging.info("No best reponders provided. Just evaluating the policy")
    else:
        for br_player, br_name in br_mapping.items():
            if name:
                name += '+'
            name += f'p{br_player}={br_name}'
            if br_name == 'straightforward':
                logging.info(f"Agent {br_player} will be straightforward")
                env_and_policy.agents[br_player] = TakeSingleActionDecorator(StraightforwardAgent(br_player, game), game.num_distinct_actions())
            elif br_name == 'tremble':
                logging.info(f"Agent {br_player} will tremble")
                env_and_policy.agents[br_player] = TremblingAgentDecorator(env_and_policy.agents[br_player], 0.05, None)
            elif br_name == 'modal':
                logging.info(f"Agent {br_player} will be modal")
                env_and_policy.agents[br_player] = ModalAgentDecorator(env_and_policy.agents[br_player])
            else: # We are loading an actual named learning agent
                if real_br:
                    raise ValueError("Can't handle more than 1 BR agent right now")
                real_br = True
                logging.info(f"Reading from agent {br_name}")
                best_response = BestResponse.objects.get(
                    checkpoint = equilibrium_solver_run_checkpoint,
                    br_player = br_player,
                    name = br_name
                )
                br_agent = load_ppo_agent(best_response) # TODO: Handle CFR trained BR agents...
                env_and_policy.agents[br_player] = br_agent 

    # RUN EVAL
    eval_output = run_eval(env_and_policy, num_samples=num_samples, report_freq=report_freq, seed=seed, compute_efficiency=compute_efficiency)

    # SAVE EVAL
    if not dry_run:
        mean_rewards = []
        for player in range(equilibrium_solver_run_checkpoint.equilibrium_solver_run.game.num_players):
            mean_rewards.append(
                pd.Series(eval_output['raw_rewards'][player]).mean()
            )
        
        eval_output = convert_pesky_np(eval_output)

        # Run NashConv only if all players are modal
        # (running NashConv against non-modal players takes a ton more memory and time)
        # TODO: is this safe to do on larger games, or will it run out of memory?
        all_modal = (len(br_mapping) == game.num_players()) and np.all([v == 'modal' for v in br_mapping.values()])
        nc = None
        if all_modal:
            worked, nc = time_bounded_run(300, nash_conv, game, env_and_policy.make_policy())
            if not worked:
                logging.info("Aborted NC calc run because time")

        Evaluation.objects.create(
            name = name,
            walltime = eval_output.pop('walltime'),
            checkpoint = equilibrium_solver_run_checkpoint,
            samples = eval_output if store_samples else [],
            mean_rewards = mean_rewards,
            best_response = best_response if real_br else None,
            nash_conv = nc,
        )
        logging.info("Saved to DB")

    print("AFTER EVAL")
    logging.info(str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)) # MiB

    return eval_output


class Command(BaseCommand):
    help = 'Evaluates a policy and saves the result'

    def add_arguments(self, parser):
        add_eval_flags(parser)
        parser.add_argument('--seed', type=int, default=EvalDefaults.DEFAULT_SEED)
        parser.add_argument('--br_mapping', type=str, default=None)

        # Needed to identify the checkpoint
        parser.add_argument('--t', type=int)
        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--run_name', type=str)

        # Needed to identify the BR along with br_name
        parser.add_argument('--dry_run', type=util.strtobool, default=0)

        add_profiling_flags(parser)

    def handle(self, *args, **options):
        setup_logging()
        opts = AttrDict(options)
        cmd = lambda: eval_command(opts.t, opts.experiment_name, opts.run_name, dict() if not opts.br_mapping else eval(opts.br_mapping), opts.dry_run, opts.seed, opts.eval_report_freq, opts.eval_num_samples, opts.eval_compute_efficiency, num_envs=opts.eval_num_envs)
        profile_cmd(cmd, opts.pprofile, opts.pprofile_file, opts.cprofile, opts.cprofile_file)
