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
import pyspiel

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

    def load_as_spiel(self):
        return pyspiel.load_game('python_clock_auction', dict(filename=self.name))

    def supply(self):
        return list(self.config['licenses'])

class Experiment(TimeStampedModel):

    name = models.TextField(unique=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ('created',)

class EquilibriumSolverRun(TimeStampedModel):
    
    experiment = models.ForeignKey(Experiment, on_delete=CASCADE)
    name = models.TextField()
    game = models.ForeignKey(Game, on_delete=CASCADE)
    config = JSONField(null=True)
    parent = models.ForeignKey('equilibriumsolverruncheckpoint', on_delete=CASCADE, null=True)
    generation = models.PositiveIntegerField(default=0)

    def walltime(self):
        return self.equilibriumsolverruncheckpoint_set.last().walltime

    def __str__(self):
        return f'{self.name} ({self.experiment})'

    def get_config_name(self):
        return self.name.split('-')[1]


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

    def get_old_eval(self):
        return self.evaluation_set.get(name='')
    
    def get_modal_eval(self):
        modal_name = ''
        for p in range(self.equilibrium_solver_run.game.num_players):
            if modal_name:
                modal_name += '+'
            modal_name += f'p{p}=modal'
        return self.evaluation_set.get(name=modal_name)

    @property
    def game(self):
        return self.equilibrium_solver_run.game

    class Meta:
        unique_together = ('equilibrium_solver_run', 't',)


class BestResponse(TimeStampedModel):

    checkpoint = models.ForeignKey(EquilibriumSolverRunCheckpoint, on_delete=CASCADE)
    br_player = models.PositiveIntegerField()
    name = models.TextField()
    walltime = models.FloatField()
    model = models.BinaryField(null=True)
    config = JSONField()
    t = models.PositiveIntegerField()

    def __str__(self):
        return f'Player {self.br_player} BR ({self.name}) to {self.checkpoint}'

    class Meta:
        unique_together = ('checkpoint', 'br_player', 'name',)

class Evaluation(TimeStampedModel):
    checkpoint = models.ForeignKey(EquilibriumSolverRunCheckpoint, on_delete=CASCADE)
    name = models.TextField()
    walltime = models.FloatField()
    samples = JSONField()
    mean_rewards = ArrayField(models.FloatField()) # For quick nash conv calcs
    best_response = models.OneToOneField(BestResponse, on_delete=CASCADE, null=True)
    nash_conv = models.FloatField(null=True)

    class Meta:
        unique_together = ('checkpoint', 'name',)

    '''
        List[Allocations],
        List[Revenue]
        List[Welfare],
        List[Rewards],
        List[Payments]
    '''

    def __str__(self):
        return f'Evaluation {self.name} for {self.checkpoint}'
