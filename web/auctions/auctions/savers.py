
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

    def __init__(self, num_players, eval_overrides, br_overrides, eq_solver_run, br_portfolio_path):
        self.num_players = num_players
        self.eval_overrides = eval_overrides
        self.br_overrides = br_overrides
        self.eq_solver_run = eq_solver_run
        self.br_portfolio_path = br_portfolio_path

    def dispatch(self, t):
        eq = self.eq_solver_run
        for player in range(self.num_players):
            # TODO: Need to change these commands!!!
            dispatch.dispatch_br_database(eq.experiment.name, eq.name, t, player, self.br_portfolio_path, overrides=self.br_overrides + " " + self.eval_overrides)
            dispatch.dispatch_eval_database(eq.experiment.name, eq.name, t, player, 'straightforward', overrides=self.eval_overrides) # Straightforward eval
        dispatch.dispatch_eval_database(eq.experiment.name, eq.name, t, None, None, overrides=self.eval_overrides)

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