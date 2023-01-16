
from auctions.models import EquilibriumSolverRunCheckpoint, BestResponse
import pickle
import logging
import open_spiel.python.examples.ubc_dispatch as dispatch

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
        eq = self.eq_solver_run
        if self.dispatch_br:
            for player in range(self.num_players):
                dispatch.dispatch_br_database(eq.experiment.name, eq.name, t, player, self.br_portfolio_path, overrides=self.br_overrides + " " + self.eval_overrides)
                if self.eval_inline: # Straightforward eval
                    from auctions.management.commands.ppo_eval import eval_command
                    logger.info(f"Running inline straightforward eval for player {player} at t={t}")
                    eval_command(t, eq.experiment.name, eq.name, 'straightforward', player, reseed=False) # TODO: This isn't reading from eval_overrides, which is a problem! You could imagine parsing it...
                else:
                    dispatch.dispatch_eval_database(eq.experiment.name, eq.name, t, player, 'straightforward', overrides=self.eval_overrides) 
        if self.eval_inline:
            from auctions.management.commands.ppo_eval import eval_command
            logger.info(f"Running inline overall eval at t={t}")
            eval_command(t, eq.experiment.name, eq.name, None, None, reseed=False) # TODO: This isn't reading from eval_overrides, which is a problem! You could imagine parsing it...
        else:
            dispatch.dispatch_eval_database(eq.experiment.name,  eq.name, t, None, None, overrides=self.eval_overrides)

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