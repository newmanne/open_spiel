import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker

plt.style.use('https://raw.githubusercontent.com/gregdeon/plots/main/style.mplstyle')
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['figure.dpi'] = 300

RESULTS_FNAME = 'cached_results_jan19.csv'
FIGURE_DIR = 'figures/comparative_statics'
FIGURE_WIDTH = 7

# names + color scheme
METRIC_PRETTY_NAMES = {
    'total_revenue': 'Revenue',
    'total_welfare': 'Welfare',
    'auction_lengths': 'Auction Length',
    'num_lotteries': 'Number of Tiebreaks',
    'unsold_licenses': 'Unsold Licenses',
}
POLICY_PRETTY_NAMES_AND_COLORS = {
    'DROP_BY_PLAYER': ('Drop by Player', 'tab:blue'),
    'DROP_BY_LICENSE': ('Drop by License', 'tab:orange'),
}

# positioning and limits
POLICY_Y_OFFSETS = {
    'DROP_BY_PLAYER': -0.12,
    'DROP_BY_LICENSE': 0.12,
}
AXIS_LIMITS = {
    'auction_lengths': (2.9, 4.1),
    'num_lotteries': (-0.1, 1.1),
    'unsold_licenses': (-0.1, 1.1),
    'total_revenue': (42.5, 44.5),
    'total_welfare': (62.5, 87.5),
}

AXIS_LIMITS.update({f'straightforward_{k}': v for k,v in AXIS_LIMITS.items()})
METRIC_PRETTY_NAMES.update({f'straightforward_{k}': v for k,v in METRIC_PRETTY_NAMES.items()})

EPSILON = 1e-6

def my_fmt(x, pos):
    if x == 0:
        return '0'
    return f'{x:.2f}'





def plot_metrics_by_game(df, metrics=None, fname='plot.png', straightforward=False):
    if metrics is None:
        metrics = ['auction_lengths', 'num_lotteries', 'unsold_licenses', 'total_revenue', 'total_welfare']
        
    if straightforward:
        metrics = [f'straightforward_{m}' for m in metrics]
    
    fig, ax_list = plt.subplots(1, len(metrics), figsize=(FIGURE_WIDTH, 1.8), sharey=True)

    game_names = np.sort(df.base_game_name.unique()).tolist()

    df_plt = df[['base_game_name', 'tiebreaking_policy']].drop_duplicates().reset_index(drop=True)
    df_plt['plt_y'] = df_plt.apply(lambda x: 1 + game_names.index(x.base_game_name) + POLICY_Y_OFFSETS[x.tiebreaking_policy], axis=1)
    df_plt['plt_color'] = df_plt.tiebreaking_policy.apply(lambda x: POLICY_PRETTY_NAMES_AND_COLORS[x][1])

    df = df.merge(df_plt, on=['base_game_name', 'tiebreaking_policy'])
    agg_dict = {**{f'min_{metric}': (metric, np.min) for metric in metrics}, **{f'max_{metric}': (metric, np.max) for metric in metrics}}
    df_ranges = df.groupby(['base_game_name', 'tiebreaking_policy'], as_index=False)[metrics].agg(**agg_dict).reset_index().merge(df_plt, on=['base_game_name', 'tiebreaking_policy'])

    for i, metric in enumerate(metrics):
        plt.sca(ax_list[i])
        plt.scatter(x=df[metric], y=df.plt_y, c=df.plt_color, s=5, zorder=10, clip_on=False)
        for _, row in df_ranges.iterrows():
            plt.plot([row[f'min_{metric}'], row[f'max_{metric}']], [row.plt_y, row.plt_y], linewidth=1, color=row.plt_color, zorder=10)
        
        # xmin, xmax = AXIS_LIMITS[metric]
        # plt.xlim(xmin, xmax)
        # if min(df[metric]) < xmin - EPSILON:
        #     print(f'WARNING: minimum {metric} of {min(df[metric])} smaller than axis limit of {xmin}')
        # if max(df[metric]) > xmax + EPSILON:
        #     print(f'WARNING: maximum {metric} of {max(df[metric])} larger than axis limit of {xmax}')
        # alternative: compute x limits depending on data
        min_metric, max_metric = min(df_ranges[f'min_{metric}']), max(df_ranges[f'max_{metric}'])
        dx = max((0.05 * (max_metric - min_metric)), 0.1)
        plt.xlim(max(0, min(df[metric]) - dx), max(df[metric]) + dx)
        plt.xlabel(METRIC_PRETTY_NAMES[metric])
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(my_fmt))


    ax_list[0].set_ylabel('Game')
    ax_list[0].set_yticks(np.arange(1, len(game_names) + 1))
    ax_list[0].set_ylim(len(game_names) + 0.5, 0.5)

    # TODO: legend. something like this, but need to get the positioning right:
    fig.subplots_adjust(bottom=0.35)
    custom_lines = [Line2D([0], [0], color=POLICY_PRETTY_NAMES_AND_COLORS[p][1], lw=2) for p in POLICY_PRETTY_NAMES_AND_COLORS]

    fig.legend(custom_lines, [POLICY_PRETTY_NAMES_AND_COLORS[p][0] for p in POLICY_PRETTY_NAMES_AND_COLORS], loc='lower center', bbox_to_anchor=(0.5,0), ncol=2)

    # plt.tight_layout()
    path = os.path.join(FIGURE_DIR, fname)
    plt.savefig(path, bbox_inches='tight')
    print(f'Saved figure to {path}')

if __name__ == '__main__':
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_FNAME)

    # main plot: 4 types, trembling on, straightforward bonus
    plot_metrics_by_game(df.query('n_types == 4 and rho == 1 and config == "cfr_port_10_extexternal_plus_linear"'), fname='comparative_statics.png')

    # varying number of types
    for num_types in [1, 2, 3, 4]:
        plot_metrics_by_game(
            df.query(f'n_types == {num_types} and rho == 1 and config == "cfr_port_10_extexternal_plus_linear"'), 
            fname=f'comparative_statics_{num_types}_types.png'
        )    
        
        plot_metrics_by_game(
            df.query(f'n_types == {num_types} and rho == 0 and config == "cfr_port_10_extexternal_plus_linear"'), 
            fname=f'comparative_statics_{num_types}_types_straightforward.png',
            straightforward=True
        )    

    # removing straightforward bonus
    plot_metrics_by_game(
        df.query(f'n_types == 4 and rho == 0 and config == "cfr_port_10_extexternal_plus_linear"'), 
        fname=f'comparative_statics_no_straightforward_bonus.png'
    )

    # removing trembling
    plot_metrics_by_game(
        df.query(f'n_types == 4 and rho == 1 and config == "cfr_port_10_extexternal_plus_linear_no_trem"'),
        fname=f'comparative_statics_no_trembling.png'
    )
