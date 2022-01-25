

"""RNN for auction games."""

import torch
from torch import nn
from absl import logging
from open_spiel.python.examples.ubc_utils import FEATURES_PER_PRODUCT, prefix_size, handcrafted_size, turn_based_size, round_index

class AuctionRNN(nn.Module):
    """
    An RNN model designed for our auction games.
    """
    def __init__(self, num_players, num_products, input_size, output_size, hidden_size=128, num_layers=1, rnn_model='lstm', nonlinearity='tanh'):
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
        super(AuctionRNN, self).__init__()

        # expect state to contain:
        # - handcrafted features
        # - turn-based game info (2 x 1-hot)
        # - current player (1-hot)
        # - budget
        # - values for each product
        # - (submitted demands, processed demands, observed demands, prices) for each product
        # (if not, we won't know how to unroll the infostate tensors)
        self.turn_based_len = turn_based_size(num_players)
        self.prefix_len = prefix_size(num_players, num_products)
        self.handcrafted_len = handcrafted_size(output_size, num_products)

        num_rounds = (input_size - self.turn_based_len - self.prefix_len - self.handcrafted_len) // (FEATURES_PER_PRODUCT * num_products)
        # confirm that this gives an integral number of rounds...
        expected_input_size = self.turn_based_len + self.handcrafted_len + self.prefix_len + FEATURES_PER_PRODUCT * num_products * num_rounds 
        assert (input_size == expected_input_size), "Expected input_size = %d, but got %d" % (expected_input_size, input_size) 

        self.num_players = num_players
        self.num_products = num_products

        input_size_per_round = self.prefix_len + FEATURES_PER_PRODUCT * num_products 
        
        if rnn_model == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_model == 'rnn':
            self.rnn = nn.RNN(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, nonlinearity=nonlinearity) 
        elif rnn_model == 'gru':
            self.rnn = nn.GRU(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError('unrecognized RNN model %s' % rnn_model) 

        self.output_layer = nn.Linear(hidden_size, output_size)

    def reshape_infostate(self, infostate_tensor):
        """
        Expand a flat infostate tensor into a per-round tensor

        Args:
        - infostate_tensor: flat list describing game infostate
    
        Outputs: tensor of shape (num rounds, model input size)
        """
        current_round = int(infostate_tensor[round_index(self.num_players)])
        infostate_tensor = infostate_tensor[self.turn_based_len + self.handcrafted_len:] # Ignore the handcrafted stuff

        # split tensor into (per-auction, per-round) features
        suffix_len_per_round = FEATURES_PER_PRODUCT * self.num_products
        prefix = torch.tensor(infostate_tensor[:self.prefix_len])
        suffix = torch.tensor(infostate_tensor[self.prefix_len:self.prefix_len + current_round * suffix_len_per_round])
        suffix_reshaped = suffix.reshape(-1, FEATURES_PER_PRODUCT * self.num_products) # (rounds, features * num_products)

        # stack
        expanded_infostate = torch.hstack([
            torch.tile(prefix, (current_round, 1)),
            suffix_reshaped
        ])
        
        return expanded_infostate
    
    def prep_batch(self, infostate_list):
        """
        Prepare a list of infostate tensors to be used as an input to the network.

        Args:
        - infostate_list: a list of infostate tensors
        
        Returns: packed sequence of input tensors
        """

        # pack into packed sequences
        packed_examples = self.pack_sequences(infostate_list)
        return packed_examples

    def forward(self, x):
        """
        Apply the model to a packed sequence x of infostates.

        Args:
        - x: packed sequence of infostates

        Outputs: (num_examples, num_actions) tensor of outputs (logit action probabilities or Q values)
        """
        # Run RNN
        rnn_outputs, _ = self.rnn(x)
        
        # Split into final output for each sequence
        padded_output, output_lens = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        last_outputs = padded_output[torch.arange(len(output_lens)), output_lens-1, :]
        
        # Apply output layer to each final output
        outputs = self.output_layer(last_outputs)

        return outputs

    def pack_sequences(self, sequences):
        """
        Turn a list of expanded infostate tensors into a packed, padded sequence. 

        Args:
        - sequences: list of tensors, each with shape (num rounds, model input size) 

        Output: model-ready packed sequence
        """
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        sequence_lengths = [len(seq) for seq in sequences]
        packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=sequence_lengths, batch_first=True, enforce_sorted=False)
        return packed_seq_batch
