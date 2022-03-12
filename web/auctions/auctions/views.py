import json
import logging
import os

from django.conf import settings
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Experiment, EquilibriumSolverRun, EquilibriumSolverRunCheckpoint, Game, BestResponse
import time

import numpy as np
import pandas as pd
from django.http import HttpResponse
import datetime
from auctions.webutils import *
from open_spiel.python.examples.ubc_sample_game_tree import sample_game_tree
from open_spiel.python.examples.ubc_decorators import TakeSingleActionDecorator
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from open_spiel.python.examples.ubc_plotting_utils import plot_all_models, parse_run

logger = logging.getLogger(__name__)

class BestResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = BestResponse
        fields = ['pk', 'checkpoint', 'br_player', 'name', 'walltime', 't']

class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = ['pk', 'name']

class ExperimentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Experiment.objects.all()
    serializer_class = ExperimentSerializer

    @action(detail=True)
    def runs(self, request, pk=None):
        experiment = self.get_object()
        runs = EquilibriumSolverRun.objects.filter(experiment=experiment)
        ser = EquilibriumSolverRunSerializer(runs, many=True, context={'request': request})
        return Response(ser.data)

class EquilibriumSolverRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = EquilibriumSolverRun
        fields = ['pk', 'name']

class EquilibriumSolverRunViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EquilibriumSolverRun.objects.all()
    serializer_class = EquilibriumSolverRunSerializer

    @action(detail=True)
    def checkpoints(self, request, pk=None):
        run = self.get_object()
        checkpoints = EquilibriumSolverRunCheckpoint.objects.filter(equilibrium_solver_run=run)
        ser = EquilibriumSolverRunCheckpointSerializer(checkpoints, many=True, context={'request': request})
        data = ser.data
        nash_conv_by_t, best_checkpoint, approx_nash_conv = find_best_checkpoint(run)
        for d in data:
            d['best'] = d['pk'] == best_checkpoint.pk
            d['ApproxNashConv'] = nash_conv_by_t[d['t']]
        
        data = sorted(data, key=lambda d: (d['best'], d['t']), reverse=True)
        return Response(data)

    @action(detail=True)
    def trajectory_plot(self, request, pk=None):
        run = self.get_object()
        ev_df = parse_run(run)
        bokeh_js = plot_all_models(ev_df, notebook=False, output_str=True)
        data = dict()
        data['bokeh_js'] = bokeh_js
        return Response(data)


class EquilibriumSolverRunCheckpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = EquilibriumSolverRunCheckpoint
        fields = ['pk', 't']

class EquilibriumSolverRunCheckpointViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = EquilibriumSolverRunCheckpoint.objects.all()
    serializer_class = EquilibriumSolverRunCheckpointSerializer

    @action(detail=True)
    def best_responses(self, request, pk=None):
        checkpoint = self.get_object()
        player = int(request.query_params['player'])
        best_responses = BestResponse.objects.filter(checkpoint=checkpoint, br_player=player)
        ser = BestResponseSerializer(best_responses, many=True, context={'request': request})
        return Response(ser.data)

class GameSerializer(serializers.ModelSerializer):
    class Meta:
        model = Game
        fields = '__all__'

class GameViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Game.objects.all()
    serializer_class = GameSerializer

    @action(detail=True)
    def experiments(self, request, pk=None):
        # What experiments has this game been part of?
        game = self.get_object()
        experiments = Experiment.objects.filter(equilibriumsolverrun__game=game).distinct()
        ser = ExperimentSerializer(experiments, many=True, context={'request': request})
        return Response(ser.data)

    @action(detail=True)
    def runs(self, request, pk=None):
        # Get runs that are for this game, possibly filtered by experiment
        game = self.get_object()
        qs = EquilibriumSolverRun.objects.filter(game=game)

        experiment_pk = request.query_params.get('experiment')
        if experiment_pk:
            qs = qs.filter(experiment=int(experiment_pk))

        ser = EquilibriumSolverRunSerializer(qs.distinct(), many=True, context={'request': request})
        return Response(ser.data)

    @action(detail=True)
    def samples(self, request, pk=None):
        game = self.get_object()

        player_0_checkpoint_pk = int(request.query_params['player_0_checkpoint_pk'])
        checkpoint = EquilibriumSolverRunCheckpoint.objects.get(pk=player_0_checkpoint_pk)
        env_and_model = db_checkpoint_loader(checkpoint)

        for player in range(game.num_players):
            checkpoint_pk = int(request.query_params[f'player_{player}_checkpoint_pk'])
            best_response_pk = request.query_params.get(f'player_{player}_br_pk')
            if best_response_pk is not None:
                best_response_pk = int(best_response_pk)
                br = BestResponse.objects.get(pk=best_response_pk)
                if br.name == 'straightforward':
                    env_and_model.agents[player] = TakeSingleActionDecorator(StraightforwardAgent(player, game.config, game.num_actions), game.num_actions)
                else:
                    env_and_model.agents[player] = load_dqn_agent(br)
            else:
                checkpoint = EquilibriumSolverRunCheckpoint.objects.get(pk=checkpoint_pk)
                env_and_model_temp = db_checkpoint_loader(checkpoint)
                env_and_model.agents[player] = env_and_model_temp.agents[player]

        data = dict()
        
        # Parse query params
        num_samples = int(request.query_params.get('num_samples', 0))
        seed = int(request.query_params.get('seed', 1234))

        # Sample
        trees = sample_game_tree(env_and_model, num_samples, seed=seed)
        data['trees'] = trees
        return Response(data)