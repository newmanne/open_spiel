import json
import logging
import os

from django.conf import settings
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Experiment
import time

import numpy as np
import pandas as pd
from django.http import HttpResponse
import datetime

logger = logging.getLogger(__name__)

class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = ['pk', 'name']


class ExperimentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Experiment.objects.all()
    serializer_class = ExperimentSerializer
