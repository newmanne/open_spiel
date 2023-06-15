
from auctions.models import EquilibriumSolverRunCheckpoint, BestResponse
import pickle
import logging
import open_spiel.python.examples.ubc_dispatch as dispatch
from open_spiel.python.examples.ubc_utils import players_not_me
from auctions.webutils import add_eval_flags
import argparse

logger = logging.getLogger(__name__)

class DBPolicySaver:

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

    def __init__(self, num_players, eval_overrides, br_overrides, eq_solver_run, br_portfolio_path, dispatch_br, eval_inline=False):
        self.num_players = num_players
        self.eval_overrides = eval_overrides
        self.br_overrides = br_overrides
        self.eq_solver_run = eq_solver_run
        self.br_portfolio_path = br_portfolio_path
        self.dispatch_br = dispatch_br
        self.eval_inline = eval_inline

    def dispatch(self, t):
        from auctions.management.commands.ppo_eval import eval_command
        # TODO: This isn't reading from eval_overrides, which is a problem! You could imagine parsing it...
        # TODO: Could modify this to leverage the game cache if that would help

        
        parser = argparse.ArgumentParser()
        add_eval_flags(parser)
        eval_args = vars(parser.parse_args(self.eval_overrides.split()))
        eval_args = {k.replace('eval_', ''):v for k,v in eval_args.items()}
        eq = self.eq_solver_run

        # Let's just do all this inline always for now
        for player in range(self.num_players):
            logger.info(f"Running inline straightforward eval for player {player} at t={t}")
            eval_command(t, eq.experiment.name, eq.name, {player: 'straightforward'}, reseed=False, **eval_args) 
            logger.info(f"Running inline trembling eval for player {player} at t={t}")
            eval_command(t, eq.experiment.name, eq.name, {p: 'tremble' for p in players_not_me(player, self.num_players)}, reseed=False, **eval_args) 
            logger.info(f"Running inline modal eval for player {player} at t={t}")
            eval_command(t, eq.experiment.name, eq.name, {player: 'modal'}, reseed=False, **eval_args) 
                    
        if self.dispatch_br:
            dispatch.dispatch_br_database(eq.experiment.name, eq.name, t, player, self.br_portfolio_path, overrides=self.br_overrides + " " + self.eval_overrides)

        # Handle evaluations
        if self.eval_inline:
            logger.info(f"Running inline overall eval at t={t}")
            eval_command(t, eq.experiment.name, eq.name, reseed=False, **eval_args) 
            logger.info(f"Running inline overall modal eval at t={t}")
            br_mapping = {p: 'modal' for p in range(self.num_players)}
            eval_command(t, eq.experiment.name, eq.name, br_mapping, reseed=False, **eval_args) 
        else:
            raise # Figure this out later or never
            dispatch.dispatch_eval_database(t, eq.experiment.name, eq.name, overrides=self.eval_overrides)

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