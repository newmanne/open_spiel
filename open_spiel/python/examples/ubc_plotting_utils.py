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
from bokeh.transform import linear_cmap, log_cmap, factor_cmap
from bokeh.palettes import Category10_10, Magma256, Spectral10, Category20_20
from bokeh.models import LinearAxis, Range1d

from bokeh.resources import CDN
from bokeh.embed import file_html

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from django.db.models import F
from auctions.models import *

PLAYER_COLORS = ['green', 'blue', 'orange']

def parse_run(run, max_t=None, expected_additional_br=1):
    '''
    expected_additional_br: In addition to straightforward, how many BRs am I expecting?
    '''
    players = list(range(run.game.num_players))

    br_evals_qs = BREvaluation.objects.filter(best_response__checkpoint__equilibrium_solver_run=run)
    if len(br_evals_qs) == 1:
        logging.warning(f"No BR found for {run}")
        return

    br_values = br_evals_qs.values(t=F('best_response__checkpoint__t'), name=F('best_response__name'), reward=F('expected_value_stats__mean'), br_player=F('best_response__br_player'))
    best_response_df = pd.DataFrame.from_records(br_values)
    best_response_df['player'] = best_response_df['br_player']

    negative_reward_br = best_response_df.query('name != "straightforward" and reward < 0')
    if len(negative_reward_br) > 0:
        logging.warning(f"Negative BR value ({negative_reward_br['reward']}) shouldn't happen. DQN should always find the drop out strategy... ")

    # Get all evaluations corresponding to the run
    evaluations = list(Evaluation.objects.filter(checkpoint__equilibrium_solver_run=run).values('mean_rewards', t=F('checkpoint__t'), length=F('samples__auction_lengths')))
    if len(evaluations) == 0:
        logging.warning(f"No evaluations found for {run}")
        return

    for i in range(len(evaluations)):
        evaluations[i]['median_length'] = pd.Series(evaluations[i]['length']).median()
        del evaluations[i]['length']

    # GOAL: Dataframe with t, reward, player, br_player, config columns
    evaluations_df = pd.DataFrame.from_records(evaluations)
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

    if ev_df.empty:
        raise ValueError("ev_df is empty!")

    if expected_additional_br is not None:
        # At least it should have an evaluation, a straightforward, and a BR. Plus 2 is b/c eval and straightforward
        # print("BEFORE", len(ev_df))
        ev_df = ev_df.groupby(['player', 't']).filter(lambda grp: len(grp) >= expected_additional_br + 2)
        # print("AFTER", len(ev_df))
        if ev_df.empty:
            raise ValueError("ev_df is empty after applying conservative filter! This likely means there are no best responses")

    logging.info("Rewards parsed. Adding regret features")

    # Regret for not having played the best response
    def get_baseline(grp):
        baseline = grp.query('br_player.isnull() and name.isnull()', engine='python')
        if len(baseline) != 1:
            t = grp["t"].unique()[0]
            raise ValueError(f"Found {len(baseline)} records for baseline! Is there are an evaluation for this timestep ({t})?")
        return baseline['reward'].iloc[0]

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
    ev_df['game'] = run.game.name

    return ev_df.sort_values('t')


def get_all_frames(experiment, truth_available=False, expected_additional_br=None):
    frames = []
    for run in tqdm(experiment.equilibriumsolverrun_set.all()):
        try:
            ev_df = parse_run(run, expected_additional_br=expected_additional_br)
            if ev_df is not None:
                frames.append(ev_df)
        except:
            logging.exception(f"Exception parsing {run}. Skipping")
    return pd.concat(frames)


def plots_to_string(plots, name=''):
    p = column(*plots)
    return file_html(p, CDN, name).strip()

def plots_to_html(plots, output_name, title='RegretPlots'):
    # Set output to static HTML file
    p = column(*plots)
    output_file(filename=output_name, title=title)
    save(p)

def plot_all_models(ev_df, notebook=True, output_name='plots.html', output_str=False, final_compare=False):
    plots = []

    # One final plot of just the NashConvs
    if final_compare:
        for game_name, grp in ev_df.groupby('game'):
            plots.append(nash_conv_plots(grp))

    for game_name, grp in ev_df.groupby('game'):
        for model, sub_df in grp.groupby('model'):
            plots.append(plot_from_df(sub_df))
            plots.append(utility_plots(sub_df))

    if notebook:
        output_notebook()
        for plot in plots:
            show(plot)
    else:
        if output_str:
            return plots_to_string(plots, 'RegretPlots')
        else:
            plots_to_html(plots, output_name)

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
    ax = sns.boxplot(x="name", y="Δ to Best Known Response", data=distance_frame.sort_values('name'))
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
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
    title = f"{model} Approximate Nash Conv"
    plot = figure(width=900, height=400, title=title)

    # add a circle renderer with a size, color, and alpha
    for p in players:
        player_color = PLAYER_COLORS[p]

        best_br_only_df = ev_df.loc[ev_df.query(f'player == {p} and br_player == {p} and name != "straightforward" and not name.isnull()', engine='python')[['t', 'PositiveRegret', 'name']].groupby('t')['PositiveRegret'].idxmax()]
        best_br_source = ColumnDataSource(best_br_only_df)
        straightforward_source = ColumnDataSource(ev_df.query(f'player == {p} and br_player == {p} and name == "straightforward"', engine='python')[['t', 'PositiveRegret']])
        overall_player_source = ColumnDataSource(ev_df.query(f'player == {p} and br_player == {p}')[['t', 'MaxPositiveRegret']].drop_duplicates())

        label = f'P{p} BR Regret'
        plot.line('t', f'PositiveRegret', source=best_br_source, legend_label=label, name=label, color=player_color)
        label = f'P{p} Straightforward Regret'
        plot.line('t', f'PositiveRegret', source=straightforward_source, legend_label=label, name=label, color=player_color, line_dash='dashed')
        label = f'P{p} Approx Regret'
        plot.line('t', f'MaxPositiveRegret', source=overall_player_source, legend_label=label, name=label, color=f'dark{player_color}', line_width=2)
        
        regret_col_name = f'p{p}_regret'
        if regret_col_name in ev_df.columns:
            label = f'P{p} True Regret'
            plot.line('t', regret_col_name, source=source, legend_label=label, name=label, color=f'dark{player_color}', line_width=2, line_dash='dotted')
        
    source = ColumnDataSource(ev_df.loc[ev_df.groupby('t')['ApproxNashConv'].idxmax()][['t', 'ApproxNashConv']]) 
    label = f'Approximate Nash Conv'
    plot.line('t', f'ApproxNashConv', source=source, legend_label=label, name=label, color='red', line_width=3)
    if 'nash_conv' in ev_df.columns:
        label = f'True Nash Conv'
        plot.line('t', f'nash_conv', source=source, legend_label=label, name=label, color='darkred', line_width=3)

    plot.legend.click_policy = "hide"
    plot.xaxis.axis_label = 'Iteration (M)'
    plot.yaxis.axis_label = 'Regret'
    plot.ray(x=[min(ev_df['t'])], y=[0], length=0, angle=0, line_width=5, color='black')

    TOOLTIPS = [
        ("Name", "$name",),
        ("(x, y)", "($data_x, $data_y)"),
    ]   
    plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    return plot

def nash_conv_plots(ev_df_ungrouped):
    game_name = ev_df_ungrouped['game'].iloc[0]
    title = f"Approximate Nash Conv {game_name}"
    colors = itertools.cycle(Category20_20) 
    plot = figure(width=900, height=400, title=title)
    for grp, ev_df in ev_df_ungrouped.groupby('model'):
        ev_df = ev_df.copy()
        model = ev_df['model'].iloc[0]

        ev_df['t'] /= 1e6 # Nicer formatting on x-axis

        # add a circle renderer with a size, color, and alpha
        source = ColumnDataSource(ev_df.loc[ev_df.groupby('t')['ApproxNashConv'].idxmax()][['t', 'ApproxNashConv']]) 
        color = next(colors)
        plot.line('t', f'ApproxNashConv', source=source, legend_label=model, name=model, color=color, line_width=3)
        plot.circle('t', f'ApproxNashConv', source=source, color=color, size=10, name=model)

    plot.legend.click_policy = "hide"
    plot.xaxis.axis_label = 'Iteration (M)'
    plot.yaxis.axis_label = 'Regret'
    plot.ray(x=[min(ev_df_ungrouped['t']) / 1e6], y=[0], length=0, angle=0, line_width=5, color='black')

    TOOLTIPS = [
        ("Name", "$name",),
        ("(x, y)", "($data_x, $data_y)"),
    ]   
    plot.add_tools(HoverTool(tooltips=TOOLTIPS))
    return plot

def utility_plots(ev_df):
    ev_df = ev_df.query('name.isnull() and br_player.isnull()', engine='python').copy()
    ev_df['t'] /= 1e6 # Nicer formatting on x-axis
    game_name = ev_df['game'].iloc[0]
    model = ev_df['model'].iloc[0]
    players = range(ev_df['num_players'].iloc[0])
    title = f"Utility {game_name} {model}"
    colors = itertools.cycle(Category20_20) 
    plot = figure(width=900, height=400, title=title)
    plot.yaxis.axis_label = 'Reward'

    # Setting the second y axis range name and range
    plot.extra_y_ranges["rounds"] = Range1d(start=0, end=50)

    # Adding the second axis to the plot.  
    plot.add_layout(LinearAxis(y_range_name="rounds", axis_label='Rounds'), 'right')

    # add a circle renderer with a size, color, and alpha
    for p in players:
        player_color = PLAYER_COLORS[p]

        source = ColumnDataSource(ev_df.query(f'player == {p}')[['t', 'reward']])
        label = f'Reward {p}'
        plot.line('t', f'reward', source=source, legend_label=label, name=label, color=player_color)
        plot.circle('t', f'reward', source=source, color=player_color, size=10, name=model)

    utility_sum = ev_df.groupby('t')['reward'].sum().to_frame('reward')
    source = ColumnDataSource(utility_sum) 
    label = f'Total Reward'
    plot.line('t', f'reward', source=source, legend_label=label, name=label, color='red', line_width=1)

    median_length = ev_df.groupby('t')['median_length'].first().to_frame('length')
    source = ColumnDataSource(median_length) 
    label = f'Median Length'
    plot.line('t', f'length', source=source, legend_label=label, name=label, color='black', line_width=1, y_range_name="rounds", line_dash='dashed')

    plot.legend.click_policy = "hide"
    plot.xaxis.axis_label = 'Iteration (M)'

    TOOLTIPS = [
        ("Name", "$name",),
        ("(x, y)", "($data_x, $data_y)"),
    ]   
    plot.add_tools(HoverTool(tooltips=TOOLTIPS))
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


def plot_embedding(df, color_col='round', reduction_method='pca', fast=False):
    df = df.copy()

    # Spaces don't play nice with the hover tooltip
    new_color_col = color_col.replace(' ', '_')
    df = df.rename(columns={color_col: new_color_col})
    color_col = new_color_col

    # Need to change newlines into <br> to have them actually break in the tooltip
    df['pretty_str'] = df['pretty_str'].apply(lambda x: x.replace('\n', '<br/>'))
    
    source = ColumnDataSource(df) # Need to drop tensors b/c of serialization issues

    plot = figure(width=900, height=900, title=f"{color_col}")

    if df.dtypes[color_col].name == 'category':
        n_cats = df[color_col].nunique()
        if n_cats > 20:
            raise ValueError("What should I do????? Too many categories")

        mapper = factor_cmap(field_name=color_col, palette=Category20_20[:n_cats], factors=list(df[color_col].unique()))
        plot.circle(f'{reduction_method}_0', f'{reduction_method}_1', size=10, color=mapper, alpha=0.3, source=source)
    else:
        mapper = linear_cmap(field_name=color_col, palette=list(reversed(Magma256)) ,low=df[color_col].min(), high=df[color_col].max())
        plot.circle(f'{reduction_method}_0', f'{reduction_method}_1', size=10, color=mapper, alpha=0.3, source=source)

    color_bar = ColorBar(color_mapper=mapper['transform'], label_standoff=12)
    plot.add_layout(color_bar, 'right')

    if not fast:
        plot.add_tools(HoverTool(tooltips=[['Infostate', '@pretty_str'],
                                        [color_col, f'@{color_col}'],
                                        ['Round', '@round'],
                                            ['Type', '@player_type'],
                                        ]))
    return plot
