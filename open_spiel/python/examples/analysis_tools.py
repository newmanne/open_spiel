# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the C++ MCCFR algorithms on Clcok Auction.

This examples calls the underlying C++ implementations via the Python bindings.
Note that there are some pure Python implementations of some of these algorithms
in python/algorithms as well.
"""

import pickle
from pathlib import Path
import sys
import pyspiel
import pandas as pd
import json
import re
import numpy as np
import time
import matplotlib.pyplot as plt

from open_spiel.python import policy
import logging

def round_probs(df):
    df['type'] = df['value'].astype(str) + 'b' + df['budget'].astype(str)
    # TODO: This double counts
    df.query('terminal and player == 0').groupby('round')['prob'].sum().plot(kind='bar')
    plt.ylabel('P(ending on round)')
    plt.xlabel("Round")


def eu(df, player):
    return df['prob'] @ df[f'Utility {player}']

def ea(df, player):
    return df['prob'] @ df[f'Allocation {player}']


def compute_expectations(x):
    player = x['player'].unique()[0]
    allocation = x['prob'] @ x[f'Allocation {player}']
    utility = x['prob'] @ x[f'Utility {player}']
    return pd.Series({'Expected Allocation': allocation, 'Expected Utility': utility})

def expected_values_over_types(df):
    return df.query('terminal').groupby(['player', 'type']).apply(compute_expectations)
