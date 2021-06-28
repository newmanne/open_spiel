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
from absl import app
from absl import flags
from pathlib import Path
import sys
import pyspiel
import pandas as pd
import json
import numpy as np


from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states
import logging

logger = logging.getLogger(__name__)
 

def sharpen_solution(f, o=None):
    if o is None:
        o = f.replace('strategy', 'reduced_strategy')
        if not o:
            o = 'reduced_strategy.csv'

    logger.info(f"Reading raw strategy from {f}")
    df = pd.read_csv(f)
    bid_cols = [c for c in df.columns if 'Bid' in c]

    terminals = df.query('terminal').copy()
    is_df = df.query('not terminal')
    is_df = is_df.set_index(['info_state', 'player'])

    prev_len = len(is_df)
    # Get rid of IS's that are never explored 
    # Of course, this will remove any IS that actually convereged to a uniform distribution. My expectation is that non-complete convergence will save us here

    action_matrix = is_df[bid_cols].values
    na_mask = ~np.isnan(action_matrix)
    non_null_row_entries = na_mask.sum(axis=1)
    only_one_unique_non_null = pd.DataFrame(action_matrix).nunique(axis=1).values == 1
    row_mask = (non_null_row_entries >= 2) & (only_one_unique_non_null)

    is_df = is_df.loc[~row_mask,:]
    
    new_len = len(is_df)
    logger.info(f"Filtered out {prev_len - new_len} infostates")
    
    # Sharpen predictions
    # Let's kill off any action you take really infrequently
    THRESH = 0.001
    action_matrix = is_df[bid_cols].values
    am = np.nan_to_num(action_matrix)
    am[am < THRESH] = 0
    # Renormalize
    row_sums = am.sum(axis=1)
    am = am / row_sums[:, np.newaxis]

    is_df.loc[:, bid_cols] = am

    is_df = is_df.reset_index()
    final_df = pd.concat([is_df, terminals], axis=0)
    final_df['terminal'] = final_df['terminal'].astype(np.bool)

    final_df = final_df.sort_values(['player', 'value', 'budget', 'round', 'my_bids', 'total_demand'])
    final_df.to_csv(o, index=False)

def main(_):
    f = FLAGS.input
    o = FLAGS.output
    sharpen_solution(f, o)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("input", "strategy.csv", "Name of the input strategy file")
    flags.DEFINE_string("output", 'reduced_strategy.csv', "Output file")

    app.run(main)

