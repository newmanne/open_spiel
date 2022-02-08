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
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

def get_all_frames(experiment_dir, truth_available=False):
    frames = []
    p = Path(experiment_dir)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    for model in subdirectories:
        try:
            ev_df = parse_rewards(os.path.join(experiment_dir, model), truth_available=truth_available)
            if ev_df is not None:
                frames.append(ev_df)
        except:
            logging.exception(f"Exception parsing {model}. Skipping")
    return pd.concat(frames)

def plot_all_models(ev_df, notebook=True, output_name='plots.html'):
    plots = []
    for model, sub_df in ev_df.groupby('model'):
        plot = plot_from_df(sub_df)
        plots.append(plot)
        
    if notebook:
        output_notebook()
        for plot in plots:
            show(plot)
    else:
        # Set output to static HTML file
        output_file(filename=output_name, title="RegretPlots")
        save(column(*plots))

def compare_best_responses(master_df):
    # Each (model, iteration) is a datapoint for each BR config
    sub_frames = []
    for _, sub_df in master_df.query('not config.isnull() and player == best_responder', engine='python').groupby(['model', 't']):
        sub_df = sub_df[['config', 'MaxPositiveRegret', 'Regret']].copy()
        sub_df['Δ to Best Known Response'] = sub_df['Regret'] - sub_df['MaxPositiveRegret']
        sub_df = sub_df[['config', 'Δ to Best Known Response']]
        sub_df['config'] = sub_df['config'].str.replace('_', ' ').str.strip()
        sub_frames.append(sub_df)
    distance_frame = pd.concat(sub_frames)

    sns.set_theme(style="ticks", palette="pastel")

    # Draw a nested boxplot 
    plt.figure(figsize=(20, 9))
    sns.boxplot(x="config", y="Δ to Best Known Response", data=distance_frame.sort_values('config'))
    sns.despine(offset=10, trim=True)

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
        return

    # 2) Group them by checkpoint name (TODO: Update for new formats without horrible re)
    pattern = re.compile(r'checkpoint_(\d+).*')
    groups = [int(re.match(pattern, os.path.basename(reward_file).split('rewards_')[1].split('.pkl')[0]).groups()[0]) for reward_file in reward_files]
    df = pd.DataFrame({'fname': reward_files, 'iteration': groups})
    records = []
    for iteration, iteration_df in df.groupby('iteration'):
        relevant_files = iteration_df['fname'].values
        for reward_file in relevant_files:
            with open(reward_file, 'rb') as f:
                rewards = pickle.load(f)
            
            # Get config and best responder. These aren't simple lookups because you didn't have foresight
            best_responder = None
            config = None
            
            straightforward_agent = rewards.get('straightforward_agent')
            if straightforward_agent is not None:
                best_responder = straightforward_agent
                config = 'Straightforward'
            else:
                best_responder = rewards['br_agent']
                br_name = rewards.get('br_name')
                if br_name is not None:
                    config = rewards['br_name'].split('br_')[1][1:] # Stupid
                    if config == '':
                        config = Path(experiment_dir).stem # TODO: Hopefully your correct this

            for player, values in rewards['rewards'].items():
                record = dict(t=iteration)
                record['reward'] = np.array(values).mean()
                record['player'] = player
                record['best_responder'] = best_responder
                record['config'] = config
                if straightforward_agent is None and record['best_responder'] == record['player'] and record['reward'] < 0:
                    logging.warning(f"Negative BR value shouldn't happen. DQN should always find the drop out strategy... Reward file={reward_file}")
                records.append(record)


    ev_df = pd.DataFrame.from_records(records)

    # Regret for not having played the best response
    def get_baseline(grp):
        return grp.query('best_responder.isnull() and config.isnull()', engine='python')['reward'].iloc[0]

    # Regret for not having played the best response
    baselines = ev_df.groupby(['t', 'player']).apply(get_baseline).reset_index().rename(columns={0: 'Baseline'})
    ev_df = ev_df.merge(baselines)
    ev_df['Regret'] = ev_df['reward'] - ev_df['Baseline']
    ev_df['PositiveRegret'] = ev_df['Regret'].clip(lower=0)


    # If player != BR player, this isn't so meaningful
    ev_df = ev_df.merge(ev_df.query('player == best_responder').groupby(['t', 'player']).apply(lambda grp: grp['PositiveRegret'].max()).reset_index().rename(columns={0: 'MaxPositiveRegret'}))
    ev_df = ev_df.merge(ev_df.groupby(['t', 'player'])['MaxPositiveRegret'].first().unstack().sum(axis='columns').reset_index().rename(columns={0:'ApproxNashConv'}))

    if truth_available:
        # TODO: This won't work until you fix it
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
    color = Category10_10.__iter__()
    title = f"{model} Approximate Nash Conv"
    plot = figure(width=900, height=400, title=title)

    # add a circle renderer with a size, color, and alpha
    PLAYER_COLORS = iter(['green', 'blue', 'purple'])
    for p in players:
        player_color = next(PLAYER_COLORS)

        best_br_only_df = ev_df.loc[ev_df.query(f'player == {p} and best_responder == {p} and config != "Straightforward" and not config.isnull()', engine='python')[['t', 'PositiveRegret', 'config']].groupby('t')['PositiveRegret'].idxmax()]
        display(best_br_only_df)
        best_br_source = ColumnDataSource(best_br_only_df)
        straightforward_source = ColumnDataSource(ev_df.query(f'player == {p} and best_responder == {p} and config == "Straightforward"', engine='python')[['t', 'PositiveRegret']])
        overall_player_source = ColumnDataSource(ev_df.query(f'player == {p} and best_responder == {p}')[['t', 'MaxPositiveRegret']].drop_duplicates())

        plot.line('t', f'PositiveRegret', source=best_br_source, legend_label=f'P{p} BR Regret', color=player_color)
        plot.line('t', f'PositiveRegret', source=straightforward_source, legend_label=f'P{p} Straightforward Regret', color=player_color, line_dash='dashed')
        plot.line('t', f'MaxPositiveRegret', source=overall_player_source, legend_label=f'P{p} Approx Regret', color=f'dark{player_color}', line_width=2)
        
        # TODO: Needs fixing
        regret_col_name = f'p{p}_regret'
        if regret_col_name in ev_df.columns:
            plot.line('t', regret_col_name, source=source, legend_label=f'P{p} True Regret', color=f'dark{player_color}', line_width=2, line_dash='dotted')
        
    source = ColumnDataSource(ev_df.loc[ev_df.groupby('t')['ApproxNashConv'].idxmax()][['t', 'ApproxNashConv']]) 
    plot.line('t', f'ApproxNashConv', source=source, legend_label=f'Approximate Nash Conv', color='red', line_width=3)
    if 'nash_conv' in ev_df.columns:
        plot.line('t', f'nash_conv', source=source, legend_label=f'True Nash Conv', color='darkred', line_width=3)

    plot.legend.click_policy = "hide"
    plot.xaxis.axis_label = 'Iteration (M)'
    plot.yaxis.axis_label = 'Regret'
    plot.ray(x=[min(ev_df['t'])], y=[0], length=0, angle=0, line_width=5, color='black')

    plot.add_tools(HoverTool())
    return plot