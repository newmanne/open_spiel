
# Understanding Iterative Combinatorial Auction Designs via Multi-Agent Reinforcement Learning

Paper: 

This repository is a fork of [OpenSpiel](https://github.com/deepmind/open_spiel) 
that was used to run experiments for the paper 
["Understanding Iterative Combinatorial Auction Designs via Multi-Agent Reinforcement Learning"](https://arxiv.org/abs/2402.19420) 
(d'Eon, Newman, Leyton-Brown, EC'24).

Some useful starting points are:
- **Clock auction implementation** (`open_spiel/python/games/`): `clock_auction.py`, along with supporting files `clock_auction_base.py`, `clock_auction_bidders.py`, `clock_auction_observer.py`, and ``clock_auction_parser.py`` 
- **Value samplers** (in `/open_spiel/python/examples/`): `pysats.py` (Python implementation of the MRVM from the [Spectrum Auction Test Suite](https://spectrumauctions.org/)) and `sats_game_sampler.py`
- **Database** (in `web/auctions/auctions/`): `models.py`, `savers.py`, and `webutils.py`
- **Experiment scripts** (in `web/auctions/auctions/management/commands/`): `cfr.py`, `ppo.py`
- **Paper experiments**: see `notebooks/readme.md`