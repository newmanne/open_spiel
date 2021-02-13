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

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "strategy.csv", "Name of the input strategy file")
flags.DEFINE_string("output", 'reduced_strategy.csv', "Output file")


def main(_):
    f = FLAGS.input
    o = FLAGS.output

    logger.info(f"Reading raw strategy from {f}")
    df = pd.read_csv(f)

    df = df.set_index(['info_state', 'player'])
    df.columns = [f'Bid {n}' for n in df.columns]

    prev_len = len(df)
    # Get rid of IS's that were never explored 
    # Of course, this will remove any IS that actually convereged to a uniform distribution. My expectation is that non-complete convergence will save us here

    # TODO: This will probably cause issues if there is only a single action you might take...  You want to verify there are >1 non-NA numbers
    df = df.loc[~((df.isna().any(axis=1)) & (df.nunique(axis=1, dropna=False) == 2)) & (df.nunique(axis=1) > 1)]
    
    new_len = len(df)
    logger.info(f"Filtered out {prev_len - new_len} infostates")
    
    # Sharpen predictions
    # Let's kill off any action you take <1% of the time
    THRESH = 0.001

    mask = df.isna()
    df = df.fillna(0)
    df[df < THRESH] = 0
    # Rebalance what remains
    df = df.div(df.sum(axis=1), axis=0)

    # Remove those 0's we added to actions you just can't play
    df = df.where(~mask, other='')

    def s(n):
        def q(x):
            lines = x.split('\n')
            if len(lines) > n:
                return lines[n]
            else:
                return ''
        return q

    df = df.reset_index()
    types = df['info_state'].apply(s(0))
    df['value'] = types.str.extract(r'v(.+)b.*').astype(np.float)
    df['budget'] = types.str.extract(r'.*b(.+)$').astype(np.float)
    df['my_bids'] = df['info_state'].apply(s(1))
    df['total_demand'] = df['info_state'].apply(s(2))
    df['round'] = df['my_bids'].apply(lambda x: 1 if x.strip() == '' else 1 + len(x.split(",")))
    df = df.sort_values(['player', 'value', 'budget', 'round', 'my_bids', 'total_demand'])
    df.to_csv(o, index=False)

    
if __name__ == "__main__":
    app.run(main)

