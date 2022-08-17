import numpy as np

# Much faster than np.random.choice, at least for our current version of numpy and our distribution over the arguments
def fast_choice(options, probs, rng=None):
    cdf = np.cumsum(probs)
    randomness = rng.rand() if rng is not None else np.random.rand()
    i = np.searchsorted(cdf / cdf[-1], randomness)
    return options[i]