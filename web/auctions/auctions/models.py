import numpy as np
import pandas as pd
from django.contrib.postgres.fields import ArrayField
from django.db.models import JSONField
from django.db import models
from django.db.models import CASCADE
from collections import defaultdict
import logging
from django_extensions.db.models import TimeStampedModel
import pickle

logger = logging.getLogger(__name__)

# TODO: Add completed field and slurm job id field for easy cancelling

def nanvalue():
    return float('nan')

class Game(TimeStampedModel):

    name = models.TextField(unique=True)
    config = JSONField()
    # Could recover from config with JSON queries, but let's make these easy to access
    num_players = models.PositiveIntegerField()
    num_actions = models.PositiveIntegerField()
    num_products = models.PositiveIntegerField()

    def __str__(self):
        return self.name

class Experiment(TimeStampedModel):

    name = models.TextField(unique=True)

    def __str__(self):
        return self.name

class EquilibriumSolverRun(TimeStampedModel):
    
    experiment = models.ForeignKey(Experiment, on_delete=CASCADE)
    name = models.TextField()
    game = models.ForeignKey(Game, on_delete=CASCADE)
    config = JSONField(null=True)

    def walltime(self):
        pass # TODO: Iterate and sum?

    def __str__(self):
        return self.name

    class Meta:
        unique_together = ('experiment', 'name',)


class EquilibriumSolverRunCheckpoint(TimeStampedModel):

    equilibrium_solver_run = models.ForeignKey(EquilibriumSolverRun, on_delete=CASCADE)
    t = models.PositiveIntegerField()
    walltime = models.FloatField()
    nash_conv = models.FloatField(null=True)
    approx_nash_conv = models.FloatField(null=True)
    policy = models.BinaryField() 

    def __str__(self):
        return f'{self.equilibrium_solver_run} Iteration {self.t}'

    def get_model(self):
        return pickle.loads(self.policy)

    class Meta:
        unique_together = ('equilibrium_solver_run', 't',)


class BestResponse(TimeStampedModel):

    checkpoint = models.ForeignKey(EquilibriumSolverRunCheckpoint, on_delete=CASCADE)
    br_player = models.PositiveIntegerField()
    name = models.TextField()
    walltime = models.FloatField()
    model = models.BinaryField()
    config = JSONField()
    t = models.PositiveIntegerField()

    def __str__(self):
        return f'Player {self.br_player} BR ({self.name}) to {self.checkpoint}'

    class Meta:
        unique_together = ('checkpoint', 'br_player', 'name',)

class BREvaluation(TimeStampedModel):

    best_response = models.ForeignKey(BestResponse, on_delete=CASCADE, null=True)
    walltime = models.FloatField()
    expected_value_cdf = ArrayField(models.FloatField(), size=101)
    expected_value_stats = JSONField()

    def __str__(self):
        return f'{self.best_response}'

class Evaluation(TimeStampedModel):
    
    walltime = models.FloatField()
    checkpoint = models.ForeignKey(EquilibriumSolverRunCheckpoint, on_delete=CASCADE)
    samples = JSONField()
    mean_rewards = ArrayField(models.FloatField()) # For quick nash conv calcs
    '''
        List[Allocations],
        List[Revenue]
        List[Welfare],
        List[Rewards],
        List[Payments]
    '''

    def __str__(self):
        return f'Evaluation for {self.checkpoint}'