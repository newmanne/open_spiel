

"""RNN for auction games."""

import torch
from torch import nn
import numpy as np
from absl import logging
from open_spiel.python.examples.ubc_utils import FEATURES_PER_PRODUCT, prefix_size, handcrafted_size, turn_based_size, round_index
from open_spiel.python.pytorch.ubc_dqn import SonnetLinear

class FlatMLP(nn.Module):
    """
    An RNN model designed for our auction games.
    """
    def __init__(self, num_players, num_products, num_types, input_size, output_size, normalizer, hidden_sizes, activate_final=False):
        """
        Initialize the model.

        Args:
        - num_players: number of bidders in the auction
        - num_products: number of products being bid on in the auction
        - output_size: number of possible actions that bidders can take (likely \prod_i(1+supply_i) )
        - hidden_size: dimension of hidden state
        - num_layers: number of recurrent layers to stack in model
        - rnn_model: type of torch model to use; options are "lstm" and "rnn"
        """
        super(FlatMLP, self).__init__()

        # expect state to contain:
        # - handcrafted features
        # - turn-based game info (2 x 1-hot)
        # - current player (1-hot)
        # - budget
        # - values for each product
        # - (submitted demands, processed demands, observed demands, prices) for each product
        # (if not, we won't know how to unroll the infostate tensors)
        self.turn_based_len = turn_based_size(num_players)
        self.prefix_len = prefix_size(num_types)
        self.handcrafted_len = handcrafted_size(output_size, num_products)
        self.normalizer = normalizer

        self.lb = self.handcrafted_len + self.turn_based_len

        self.num_players = num_players
        self.num_products = num_products
        
        self._layers = []
        # Hidden layers
        for size in hidden_sizes:
            self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
            input_size = size
        # Output layer

        self._layers.append(
            SonnetLinear(
                in_size=input_size,
                out_size=output_size,
                activate_relu=activate_final))

        self.model = nn.ModuleList(self._layers)


    def reshape_infostate(self, infostate_tensor):
        # MLP doesn't need to reshape infostates: just use flat tensor
        infostate_tensor = torch.tensor(infostate_tensor) / self.normalizer[:len(infostate_tensor)]
        return infostate_tensor[self.lb:]

    def prep_batch(self, infostate_list):        
        """
        Prepare a list of infostate tensors to be used as an input to the network.

        Args:
        - infostate_list: a list of infostate tensors, each with shape (num_features)
        
        Returns: (num_examples, num_features) tensor of features
        """
        return torch.vstack(infostate_list)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def get_last_layer(self):
        return self._layers[-1]

