from django.core.management.base import BaseCommand
from open_spiel.python.examples.ppo_eval import run_eval, EvalDefaults
from open_spiel.python.examples.ubc_utils import series_to_quantiles, fix_seeds, players_not_me, time_bounded_run
from open_spiel.python.examples.ubc_cma import analyze_samples, get_modal_nash_conv_new_rho
import logging
from auctions.models import *
from auctions.webutils import *
from open_spiel.python.examples.ubc_decorators import TakeSingleActionDecorator, TremblingAgentDecorator, ModalAgentDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from distutils import util
from open_spiel.python.algorithms.exploitability import nash_conv
import os, psutil
import humanize
from compress_pickle import dumps, loads
import wandb
from collections.abc import Iterable


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

def eval_command(t, experiment_name, run_name, br_mapping=None, dry_run=False, seed=EvalDefaults.DEFAULT_SEED, report_freq=EvalDefaults.DEFAULT_REPORT_FREQ, num_samples=EvalDefaults.DEFAULT_NUM_SAMPLES, compute_efficiency=False, num_envs=EvalDefaults.DEFAULT_NUM_ENVS, restrict_to_heuristics=EvalDefaults.DEFAULT_RESTRICT_TO_HEURISTICS, reseed=True, store_samples=True, use_wandb=False, game=None, policy=None, rho=0):
    '''br_mapping is a dict with keys of player_ids mapping to br_names'''

    if br_mapping is None:
        br_mapping = dict()
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

    ### To save on memory, allow reusing an existing game and policy (e.g., for an inline eval). Those tabular algorithms get big! Plus, you can reuse the game caches this way. Note that we will modify the AGENTS, so we don't pass env_and_policy. But the POLICY object should not be altered
    if game is not None and policy is not None:
        env_and_policy = make_env_and_policy(game, dict(equilibrium_solver_run_checkpoint.equilibrium_solver_run.config), env_params=env_params)
        for agent in env_and_policy.agents:
            agent.policy = policy
    else:
        env_and_policy = ppo_db_checkpoint_loader(equilibrium_solver_run_checkpoint, env_params=env_params)
        game = env_and_policy.game

    # Modify rho if necessary
    if rho is None or rho == game.auction_params.sor_bid_bonus_rho:
        logger.info(f"Running evaluation with game's original rho value ({game.auction_params.sor_bid_bonus_rho})")
    else:
        logger.info(f"Running evaluation on modified game with rho = {rho}")
        game_rho = game.load_copy()
        game_rho.auction_params.sor_bid_bonus_rho = rho

        env_and_policy.game = game_rho
        game = game_rho

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

    # Run NashConv only if all players are modal or all players are straightforward
    # (running NashConv against non-modal players takes a ton more memory and time)
    # TODO: This is not necessarily safe and has a tendency to run out of memory and crash the program :(. Can we isolate it better? 
    all_modal = (len(br_mapping) == game.num_players()) and np.all([v == 'modal' for v in br_mapping.values()])
    all_straightforward = (len(br_mapping) == game.num_players()) and np.all([v == 'straightforward' for v in br_mapping.values()])
    nc, nash_conv_player_improvements, nash_conv_runtime, hc, heuristic_conv_player_improvements, heuristic_conv_runtime = None, None, None, None, None, None
    if all_modal or all_straightforward:
        if not restrict_to_heuristics:
            # compute nash_conv
            nc_time_limit_seconds = 300
            logging.info(f"Computing NC for up to {nc_time_limit_seconds} seconds")
            worked, nash_conv_runtime, res = time_bounded_run(nc_time_limit_seconds, nash_conv, game, env_and_policy.make_policy(), return_only_nash_conv=False, restrict_to_heuristics=False)
            if worked:
                (nc, nash_conv_player_improvements, br_policies) = res
                if not dry_run:
                    for player in range(game.num_players()):
                        # delete existing best response, if it exists
                        try:
                            br = BestResponse.objects.get(
                                checkpoint = equilibrium_solver_run_checkpoint,
                                br_player = player,
                                name = 'NashConv'
                            )
                            br.delete()
                        except BestResponse.DoesNotExist:
                            pass

                        br_policies[player].delete_fields_for_saving()
                        BestResponse.objects.create(
                            checkpoint = equilibrium_solver_run_checkpoint,
                            br_player = player,
                            walltime = nash_conv_runtime,
                            model = dumps(br_policies[player], compression='gzip'),
                            config = dict(),
                            name = 'NashConv',
                            t = 0,
                        )

                nash_conv_player_improvements = list(nash_conv_player_improvements) # For db saving
                logging.info(f"NC results: NashConv:{nc} PlayerImprovements: {nash_conv_player_improvements}. Took {nash_conv_runtime:.2f} s")
            else:
                logging.info("Aborted NC calc run because time")

        # compute heuristic_conv
        logging.info(f"Computing HC for up to {nc_time_limit_seconds} seconds")
        worked, heuristic_conv_runtime, res = time_bounded_run(nc_time_limit_seconds, nash_conv, game, env_and_policy.make_policy(), return_only_nash_conv=False, restrict_to_heuristics=True)
        if worked:
            (hc, heuristic_conv_player_improvements, br_policies) = res
            if not dry_run:
                for player in range(game.num_players()):
                    # delete existing best response, if it exists
                    try:
                        br = BestResponse.objects.get(
                            checkpoint = equilibrium_solver_run_checkpoint,
                            br_player = player,
                            name = 'HeuristicConv'
                        )
                        br.delete()
                    except BestResponse.DoesNotExist:
                        pass

                    br_policies[player].delete_fields_for_saving()
                    BestResponse.objects.create(
                        checkpoint = equilibrium_solver_run_checkpoint,
                        br_player = player,
                        walltime = heuristic_conv_runtime,
                        model = dumps(br_policies[player], compression='gzip'),
                        config = dict(),
                        name = 'HeuristicConv',
                        t = 0,
                    )

            heuristic_conv_player_improvements = list(heuristic_conv_player_improvements) # For db saving
            logging.info(f"HC results: HeuristicConv: {hc} PlayerImprovements: {heuristic_conv_player_improvements}. Took {heuristic_conv_runtime:.2f} s")
        else:
            logging.info("Aborted HC calc run because time")

    # for modal: if we solved a modified game, also compute NashConv on the original game
    validation_info = None
    # if all_modal:
    #     rho = game.auction_params.sor_bid_bonus_rho
    #     validation_info = {}
    #     if rho == 0:
    #         # for base game, write nash_conv and heuristic_conv as computed above
    #         logging.info(f"Game had no indifference-breaking; using previously computed NashConvs")
    #         if not restrict_to_heuristics:
    #             validation_info.update({
    #                 'rho_0_nash_conv': nc,
    #                 'rho_0_nash_conv_runtime': nash_conv_runtime,
    #                 'rho_0_nash_conv_player_improvements': nash_conv_player_improvements,
    #             })
    #         validation_info.update({
    #             'rho_0_heuristic_conv': hc,
    #             'rho_0_heuristic_conv_runtime': heuristic_conv_runtime,
    #             'rho_0_heuristic_conv_player_improvements': heuristic_conv_player_improvements,
    #         })
    #     else:
    #         # otherwise, compute nash_conv and heuristic_conv for the rho=0 game
    #         # note that we need to make a copy of the game to avoid modifying the original one
    #         game_db = equilibrium_solver_run_checkpoint.equilibrium_solver_run.game
    #         if not restrict_to_heuristics:
    #             logging.info(f"Computing NashConv on game with rho=0 for up to {nc_time_limit_seconds} seconds")
    #             nc_rho_0_runtime, nc_rho_0_res = get_modal_nash_conv_new_rho(game_db.load_as_spiel(), env_and_policy.make_policy(), c, rho=0, return_only_nash_conv=False, restrict_to_heuristics=False, time_limit_seconds=nc_time_limit_seconds)
    #             nc_rho_0, nc_rho_0_player_improvements, _ = nc_rho_0_res if nc_rho_0_res is not None else (None, None, None)
    #             validation_info.update({
    #                 'rho_0_nash_conv': nc_rho_0,
    #                 'rho_0_nash_conv_runtime': nc_rho_0_runtime,
    #                 'rho_0_nash_conv_player_improvements': nc_rho_0_player_improvements,
    #             })

    #         logging.info(f"Computing HeuristicConv on game with rho=0 for up to {nc_time_limit_seconds} seconds")
    #         hc_rho_0_runtime, hc_rho_0_res = get_modal_nash_conv_new_rho(game_db.load_as_spiel(), env_and_policy.make_policy(), c, rho=0, return_only_nash_conv=False, restrict_to_heuristics=True, time_limit_seconds=nc_time_limit_seconds)
    #         hc_rho_0_res, hc_rho_0_player_improvements, _ = hc_rho_0_res if hc_rho_0_res is not None else (None, None, None)
    #         validation_info.update({                
    #             'rho_0_heuristic_conv': hc_rho_0_res,
    #             'rho_0_heuristic_conv_runtime': hc_rho_0_runtime, 
    #             'rho_0_heuristic_conv_player_improvements': hc_rho_0_player_improvements,
    #         })

    # SAVE EVAL
    if not dry_run:
        mean_rewards = []
        for player in range(equilibrium_solver_run_checkpoint.equilibrium_solver_run.game.num_players):
            mean_rewards.append(
                pd.Series(eval_output['raw_rewards'][player]).mean()
            )
        
        eval_output = convert_pesky_np(eval_output)
        validation_info = convert_pesky_np(validation_info)

        Evaluation.objects.create(
            name = name,
            walltime = eval_output.pop('walltime'),
            checkpoint = equilibrium_solver_run_checkpoint,
            samples = eval_output if store_samples else [],
            mean_rewards = mean_rewards,
            best_response = best_response if real_br else None,
            nash_conv = nc,
            nash_conv_player_improvements = nash_conv_player_improvements,
            nash_conv_runtime = nash_conv_runtime,
            heuristic_conv = hc,
            heuristic_conv_player_improvements = heuristic_conv_player_improvements,
            heuristic_conv_runtime = heuristic_conv_runtime,
            validation_info = validation_info,
        )
        logging.info("Saved to DB")

        if use_wandb:
            # FIXME: seems to be inconsistent at logging to wandb 
            wandb_data = {
                **analyze_samples(eval_output, game, restrict_to_wandb=True),
                **{f'mean_reward_{p}': v for p, v in enumerate(mean_rewards)}
            }
            if hc is not None:
                wandb_data.update({
                    'heuristic_conv': hc,
                    'heuristic_conv_runtime': heuristic_conv_runtime,
                    **{f'heuristic_conv_player_improvements_{p}': v for p, v in enumerate(heuristic_conv_player_improvements)}
                })
            if nc is not None:
                wandb_data.update({
                    'nash_conv': nc,
                    'nash_conv_runtime': nash_conv_runtime,
                    **{f'nash_conv_player_improvements_{p}': v for p, v in enumerate(nash_conv_player_improvements)}
                })
            if validation_info is not None:
                for k, v in validation_info.items():
                    if isinstance(v, Iterable):
                        wandb_data.update({f'{k}_{i}': vi for i, vi in enumerate(v)})
                    else: 
                        wandb_data[k] = v

            wandb.log({f'eval_{"base" if name == "" else name}/{k}': v for k, v in wandb_data.items()})

    logging.info("AFTER EVAL")
    
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
