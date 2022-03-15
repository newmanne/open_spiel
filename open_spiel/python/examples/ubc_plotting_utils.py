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
import tempfile
import subprocess

from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.algorithms.exploitability import nash_conv, best_response

import bokeh
from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column
from bokeh.io import output_notebook
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, LogColorMapper, LinearColorMapper
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import Category10_10, Magma256

from bokeh.resources import CDN
from bokeh.embed import file_html

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from django.db.models import F
from auctions.models import *

def parse_run(run, max_t=None):
    players = list(range(run.game.num_players))

    br_evals_qs = BREvaluation.objects.filter(best_response__checkpoint__equilibrium_solver_run=run)
    if len(br_evals_qs) == 0:
        logging.warning(f"No BR found for {run}")
        return

    br_values = br_evals_qs.values(t=F('best_response__checkpoint__t'), name=F('best_response__name'), reward=F('expected_value_stats__mean'), br_player=F('best_response__br_player'))
    best_response_df = pd.DataFrame.from_records(br_values)
    best_response_df['player'] = best_response_df['br_player']

    negative_reward_br = best_response_df.query('name != "straightforward" and reward < 0')
    if len(negative_reward_br) > 0:
        logging.warning(f"Negative BR value ({negative_reward_br['reward']}) shouldn't happen. DQN should always find the drop out strategy... ")

    # Get all evaluations corresponding to the run
    evaluations = Evaluation.objects.filter(checkpoint__equilibrium_solver_run=run).values('mean_rewards', t=F('checkpoint__t'))
    if len(evaluations) == 0:
        logging.warning(f"No evaluations found for {run}")
        return

    # GOAL: Dataframe with t, reward, player, br_player, config columns
    eval_values = evaluations
    evaluations_df = pd.DataFrame.from_records(eval_values)
    evaluations_df['br_player'] = None
    evaluations_df['name'] = None
    frames = []
    for player_id in players:
        frame = evaluations_df.copy()
        frame['reward'] = frame['mean_rewards'].apply(lambda x: x[player_id])
        frame['player'] = player_id
        del frame['mean_rewards']
        frames.append(frame)
    ev_df = pd.concat((best_response_df, *frames))

    logging.info("Rewards parsed. Adding regret features")

    # Regret for not having played the best response
    def get_baseline(grp):
        return grp.query('br_player.isnull() and name.isnull()', engine='python')['reward'].iloc[0]

    # Regret for not having played the best response
    baselines = ev_df.groupby(['t', 'player']).apply(get_baseline).reset_index().rename(columns={0: 'Baseline'})
    ev_df = ev_df.merge(baselines)
    ev_df['Regret'] = ev_df['reward'] - ev_df['Baseline']
    ev_df['PositiveRegret'] = ev_df['Regret'].clip(lower=0)

    if max_t is not None:
        ev_df = ev_df.query(f't <= {max_t}')

    # If player != BR player, this isn't so meaningful
    ev_df = ev_df.merge(ev_df.query('player == br_player').groupby(['t', 'player']).apply(lambda grp: grp['PositiveRegret'].max()).reset_index().rename(columns={0: 'MaxPositiveRegret'}))
    ev_df = ev_df.merge(ev_df.groupby(['t', 'player'])['MaxPositiveRegret'].first().unstack().sum(axis='columns').reset_index().rename(columns={0:'ApproxNashConv'}))

    # if truth_available:
    #     # TODO: This won't work until you fix it
    #     true_br_df = make_true_br_df(experiment_dir)
    #     ev_df = ev_df.merge(true_br_df, on='t')

    ev_df['num_players'] = run.game.num_players
    ev_df['model'] = run.name

    return ev_df.sort_values('t')


def get_all_frames(experiment, truth_available=False):
    frames = []
    for run in tqdm(experiment.equilibriumsolverrun_set.all()):
        try:
            ev_df = parse_run(run)
            if ev_df is not None:
                frames.append(ev_df)
        except:
            logging.exception(f"Exception parsing {run}. Skipping")
    return pd.concat(frames)


def plots_to_string(plots, name=''):
    p = column(*plots)
    return file_html(p, CDN, name).strip()

def plot_all_models(ev_df, notebook=True, output_name='plots.html', output_str=False):
    plots = []
    for model, sub_df in ev_df.groupby('model'):
        plot = plot_from_df(sub_df)
        plots.append(plot)
        
    if notebook:
        output_notebook()
        for plot in plots:
            show(plot)
    else:
        if output_str:
            return plots_to_string(plots, 'RegretPlots')
        else:
            # Set output to static HTML file
            p = column(*plots)
            output_file(filename=output_name, title="RegretPlots")
            save(p)

def compare_best_responses(master_df):
    # Each (model, iteration) is a datapoint for each BR config
    sub_frames = []
    for _, sub_df in master_df.query('not name.isnull() and player == br_player', engine='python').groupby(['model', 't']):
        sub_df = sub_df[['name', 'MaxPositiveRegret', 'Regret']].copy()
        sub_df['Δ to Best Known Response'] = sub_df['Regret'] - sub_df['MaxPositiveRegret']
        sub_df = sub_df[['name', 'Δ to Best Known Response']]
        sub_df['name'] = sub_df['name'].str.replace('_', ' ').str.strip()
        sub_frames.append(sub_df)
    distance_frame = pd.concat(sub_frames)

    sns.set_theme(style="ticks", palette="pastel", font_scale=2)

    # Draw a nested boxplot 
    fig = plt.figure(figsize=(30, 9))
    # sns.ecdfplot(hue="config", x="Δ to Best Known Response", data=distance_frame.sort_values('config'))
    sns.boxplot(x="name", y="Δ to Best Known Response", data=distance_frame.sort_values('name'))
    # sns.despine(offset=10, trim=True)
    return fig

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

        best_br_only_df = ev_df.loc[ev_df.query(f'player == {p} and br_player == {p} and name != "straightforward" and not name.isnull()', engine='python')[['t', 'PositiveRegret', 'name']].groupby('t')['PositiveRegret'].idxmax()]
        # display(best_br_only_df)
        best_br_source = ColumnDataSource(best_br_only_df)
        straightforward_source = ColumnDataSource(ev_df.query(f'player == {p} and br_player == {p} and name == "straightforward"', engine='python')[['t', 'PositiveRegret']])
        overall_player_source = ColumnDataSource(ev_df.query(f'player == {p} and br_player == {p}')[['t', 'MaxPositiveRegret']].drop_duplicates())

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

def special_save_fig(fig, file_name, fmt=None, dpi=300, tight=True):
    """Save a Matplotlib figure as EPS/PNG/PDF to the given path and trim it.
    """
    if not fmt:
        fmt = file_name.strip().split('.')[-1]

    if fmt not in ['eps', 'png', 'pdf']:
        raise ValueError('unsupported format: %s' % (fmt,))

    extension = '.%s' % (fmt,)
    if not file_name.endswith(extension):
        file_name += extension

    file_name = os.path.abspath(file_name)

    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_name = tmp_file.name + extension

    # save figure
    if tight:
        fig.savefig(tmp_name, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(tmp_name, dpi=dpi)

    #trim it
    if fmt == 'eps':
        subprocess.call('epstool --bbox --copy %s %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'png':
        subprocess.call('convert %s -trim %s' %
                        (tmp_name, file_name), shell=True)
    elif fmt == 'pdf':
        subprocess.call('pdfcrop %s %s' % (tmp_name, file_name), shell=True)


def plot_embedding(df, color_col='round'):
    df = df.copy()

    # Spaces don't play nice with the hover tooltip
    new_color_col = color_col.replace(' ', '_')
    df = df.rename(columns={color_col: new_color_col})
    color_col = new_color_col

    # Need to change newlines into <br> to have them actually break in the tooltip
    df['pretty_str'] = df['pretty_str'].apply(lambda x: x.replace('\n', '<br/>'))
    
    source = ColumnDataSource(df) # Need to drop tensors b/c of serialization issues

    plot = figure(width=900, height=400, title=f"{color_col}")

    # add a circle renderer with a size, color, and alpha
    mapper = linear_cmap(field_name=color_col, palette=list(reversed(Magma256)) ,low=df[color_col].min(), high=df[color_col].max())
#     mapper = log_cmap(field_name=f'Prob_{action_num}', palette="Magma256" ,low=1e-9, high=q[action_cols].values.max())
    plot.circle('pca_0', 'pca_1', size=10, color=mapper, alpha=0.3, source=source)


    plot.add_tools(HoverTool(tooltips=[['Infostate', '@pretty_str'],
                                       [color_col, f'@{color_col}'],
                                       ['Round', '@round'],
                                      ]))
    color_bar = ColorBar(color_mapper=mapper['transform'], label_standoff=12)
    plot.add_layout(color_bar, 'right')
    return plot
