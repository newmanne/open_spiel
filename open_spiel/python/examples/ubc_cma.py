import numpy as np
import pandas as pd
from collections import Counter

def analyze_checkpoint(checkpoint):
    record = dict()
    samples = checkpoint.evaluation.samples
    welfares = np.zeros(len(samples['rewards']['0']))
    revenues = np.zeros(len(samples['rewards']['0']))
    for player in range(checkpoint.equilibrium_solver_run.game.num_players):
        player = str(player)
        rewards = pd.Series(samples['rewards'][player])
        payments = pd.Series(samples['payments'][player])
        record[f'p{player}_utility'] = rewards.mean()
        record[f'p{player}_payment'] = payments.mean()
        welfares += rewards + payments
        revenues += payments
    record['total_welfare'] = welfares.mean()
    record['total_revenue'] = revenues.mean()
    record['auction_lengths'] = np.array(samples['auction_lengths']).mean()

    arr = np.array(samples['allocations']['0']).astype(int) # TODO: only player 0's allocation.
    c = Counter(tuple(map(tuple, arr)))
    record['common_allocations'] = c.most_common(5)
    return record