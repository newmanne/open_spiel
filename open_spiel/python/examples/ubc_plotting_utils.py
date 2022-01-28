import sys
import glob
import pandas as pd
import os
import seaborn as sns

from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict
import pickle
import re
import json
import numpy as np 
import pandas as pd
from absl import logging

from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.examples.ubc_nfsp_example import policy_from_checkpoint
from open_spiel.python.algorithms.exploitability import nash_conv, best_response

import bokeh
from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column
from bokeh.io import output_notebook
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, LogColorMapper, LinearColorMapper
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import Category10_10

def get_checkpoints(d):
    checkpoints = glob.glob(f'{d}/solving_checkpoints/*.pkl')
    checkpoints = [os.path.basename(c).replace('.pkl', '') for c in checkpoints if 'latest' not in c]
    times = [int(c.split('_')[1]) for c in checkpoints]
    return zip(checkpoints, times)

def make_true_br_df(d):
    # TODO: This is highly specialized to 2 players and could easily not be
    df_path = f'{d}/true_br_df.csv'
    if os.path.exists(df_path): # Quick and dirty cache
        print("Found copy, reading")
        return pd.read_csv(df_path)
    
    records = []
    for checkpoint, time in tqdm(get_checkpoints(d)):
        env_and_model = policy_from_checkpoint(d, checkpoint) 
        policy = env_and_model.nfsp_policies
        game = env_and_model.game

        p0 = best_response(game, policy, 0)
        p1 = best_response(game, policy, 1)

        record = {
            't': time,
            'p0_br': p0['best_response_value'],
            'p0_policy': p0['on_policy_value'],
            'p1_br': p1['best_response_value'],
            'p1_policy': p1['on_policy_value'],
            'p0_regret': p0['best_response_value'] - p0['on_policy_value'],
            'p1_regret': p1['best_response_value'] - p1['on_policy_value'],
            'nash_conv': (p0['best_response_value'] - p0['on_policy_value']) + (p1['best_response_value'] - p1['on_policy_value'])
        }
        records.append(record)

    true_br_df = pd.DataFrame.from_records(records)
    true_br_df.to_csv(f'{d}/true_br_df.csv', index=False) 
    return true_br_df

def parse_rewards(experiment_dir, truth_available=False):
    # 0) Read game
    game = smart_load_sequential_game('clock_auction', dict(filename=f'{experiment_dir}/game.json'))
    players = range(game.num_players())

    # 1) Get all reward files
    reward_files = glob.glob(experiment_dir + f'/evaluations/rewards_*.pkl')

    if len(reward_files) == 0:
        logging.warning(f"No files found for {experiment_dir}")

    # 2) Group them by checkpoint name (TODO: Update for new formats without horrible re)
    pattern = re.compile(r'checkpoint_(\d+).*')
    groups = [int(re.match(pattern, os.path.basename(reward_file).split('rewards_')[1].split('.pkl')[0]).groups()[0]) for reward_file in reward_files]
    df = pd.DataFrame({'fname': reward_files, 'iteration': groups})
    records = []
    for iteration, iteration_df in df.groupby('iteration'):
        record = dict(t=iteration)
        
        relevant_files = iteration_df['fname'].values
        for reward_file in relevant_files:
            
            with open(reward_file, 'rb') as f:
                rewards = pickle.load(f)
                
            straightforward_agent = rewards.get('straightforward_agent')
            if straightforward_agent is not None:
                for player, values in rewards['rewards'].items():
                    key = f'{player}_straightforward{straightforward_agent}'
                    record[key] = np.array(values).mean()
            else:
                for player, values in rewards['rewards'].items():
                    br_agent = rewards['br_agent']
                    if br_agent is None:
                        key = str(player)
                    else:
                        key = f'{player}_{br_agent}'
                    record[key] = np.array(values).mean()
                    if key == f'{br_agent}_{br_agent}' and record[key] < 0:
                        logging.warning("Negative BR value shouldn't happen. DQN should always find the drop out strategy...")
        
        records.append(record)
        
    ev_df = pd.DataFrame.from_records(records)

    # Regret for not having played the best response
    for p in players:
        ev_df[f"BRRegret{p}"] = ev_df[f"{p}_{p}"] - ev_df[f"{p}"]
        ev_df[f'StraightforwardRegret{p}'] = ev_df[f"{p}_straightforward{p}"] - ev_df[f"{p}"]
        ev_df[f'Regret{p}'] = ev_df[[f'BRRegret{p}', f'StraightforwardRegret{p}']].max(axis=1)
    regret_cols = [f'Regret{p}' for p in players] 
    ev_df['ApproxNashConv'] = ev_df[regret_cols].clip(lower=0).sum(axis='columns')
    if truth_available:
        true_br_df = make_true_br_df(experiment_dir)
        ev_df = ev_df.merge(true_br_df, on='t')

    ev_df['num_players'] = game.num_players()
    ev_df['model'] = os.path.basename(experiment_dir)

    return ev_df

def plot_from_df(ev_df):
    # curdoc().clear()

    ev_df = ev_df.copy() # Don't want to change actual frame
    players = range(ev_df['num_players'].iloc[0])
    model = ev_df['model'].iloc[0]

    ev_df['t'] /= 1e6 # Nicer formatting on x-axis
    source = ColumnDataSource(ev_df) # Need to drop tensors b/c of serialization issues
        
    color = Category10_10.__iter__()
    title = f"{model} Approximate Nash Conv"
    plot = figure(width=900, height=400, title=title)

    # add a circle renderer with a size, color, and alpha
    PLAYER_COLORS = iter(['green', 'blue', 'purple'])
    for p in players:
        player_color = next(PLAYER_COLORS)
        plot.line('t', f'BRRegret{p}', source=source, legend_label=f'P{p} BR Regret', color=player_color)
        plot.line('t', f'StraightforwardRegret{p}', source=source, legend_label=f'P{p} Straightforward Regret', color=player_color, line_dash='dashed')
        plot.line('t', f'Regret{p}', source=source, legend_label=f'P{p} Approx Regret', color=f'dark{player_color}', line_width=2)
        
        regret_col_name = f'p{p}_regret'
        if regret_col_name in ev_df.columns:
            plot.line('t', regret_col_name, source=source, legend_label=f'P{p} True Regret', color=f'dark{player_color}', line_width=2, line_dash='dotted')
        
    plot.line('t', f'ApproxNashConv', source=source, legend_label=f'Approximate Nash Conv', color='red', line_width=3)
    if 'nash_conv' in ev_df.columns:
        plot.line('t', f'nash_conv', source=source, legend_label=f'True Nash Conv', color='darkred', line_width=3)

    plot.legend.click_policy = "hide"
    plot.xaxis.axis_label = 'Iteration (M)'
    plot.yaxis.axis_label = 'Regret'
    plot.ray(x=[min(ev_df['t'])], y=[0], length=0, angle=0, line_width=5, color='black')

    plot.add_tools(HoverTool())
    return plot