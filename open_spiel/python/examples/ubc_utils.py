import pyspiel
from absl import logging

CLOCK_AUCTION = 'clock_auction'

def smart_load_sequential_game(game_name, game_parameters=dict()):
    # Stupid special case our own game because loading it twice takes time
    if game_name != CLOCK_AUCTION:
        game = pyspiel.load_game(game_name, game_parameters)
        game_type = game.get_type()

    if game_name == CLOCK_AUCTION or game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        logging.warn("%s is not turn-based. Trying to reload game as turn-based.", game_name)
        game = pyspiel.load_game_as_turn_based(game_name, game_parameters)

    game_type = game.get_type()

    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must be sequential, not {}".format(game_type.dynamics))

    return game