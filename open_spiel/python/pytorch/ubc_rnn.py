

"""RNN for auction games."""

import torch
from torch import nn

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
        # - turn-based game info (2 x 1-hot)
        # - current player (1-hot)
        # - budget
        # - values for each product
        # - (submitted demands, processed demands, observed demands, prices) for each product
        # (if not, we won't know how to unroll the infostate tensors)
        features_per_product = 4
        num_rounds = (input_size - (3 * num_players + 1 + num_products)) // (features_per_product * num_products)
        # confirm that this gives an integral number of rounds...
        expected_input_size = 3 * num_players + 1 + num_products + features_per_product * num_products * num_rounds 
        assert (input_size == expected_input_size), "Expected input_size = %d, but got %d" % (expected_input_size, input_size) 

        self.num_players = num_players
        self.num_products = num_products

        input_size_per_round = 3 * num_players + 1 + (features_per_product + 1) * num_products 
        
        if rnn_model == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_model == 'rnn':
            self.rnn = nn.RNN(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, nonlinearity=nonlinearity) 
        elif rnn_model == 'gru':
            self.rnn = nn.GRU(input_size=input_size_per_round, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError('unrecognized RNN model %s' % rnn_model) 

        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Apply the model to a tensor x of infostates.

        Args:
        - x: (num_examples, infostate_size) tensor of infostates

        Outputs: (num_examples, num_actions) tensor of outputs (logit action probabilities or Q values)
        """

        # Expand infostates into packed sequences
        # TODO: this seems like a slow hack to avoid changing the SL/RL buffers...
        infostates = list(x)
        infostate_sequences = [self.expand_infostate_tensor(t) for t in infostates]
        packed_input = self.pack_sequences(infostate_sequences)

        # then, run RNN
        rnn_outputs, _ = self.rnn(packed_input)
        padded_output, output_lens = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        last_outputs = torch.cat([padded_output[e, i-1, :].unsqueeze(0) for e, i in enumerate(output_lens)])
        outputs = self.output_layer(last_outputs)

        return outputs

    def expand_infostate_tensor(self, infostate_tensor):
        """
        Expand a flat infostate tensor into a per-round tensor

        Args:
        - infostate_tensor: flat tensor describing game infostate
    
        Outputs: tensor of shape (num rounds, model input size)
        """
        offset = 0
        
        prefix_len = 3 * self.num_players + 1 + self.num_products
        suffix_len_per_round = 4 * self.num_products
        
        # split tensor into (per-auction, per-round) features
        prefix = infostate_tensor[:prefix_len]
        suffix = infostate_tensor[prefix_len:]
        suffix_reshaped = suffix.reshape(4, -1, self.num_products) # (features, rounds, products)
        suffix_expanded = torch.permute(suffix_reshaped, (1, 0, 2)).reshape(-1, 4 * self.num_products) # (rounds, features x products)

        # hack: look at non-zero prices to figure out which rounds actually happened 
        current_round = (suffix_expanded[:, -1] > 0).sum()

        # stack
        expanded_infostate = torch.hstack([
            torch.tile(prefix, (current_round, 1)),
            suffix_expanded[:current_round, :]
        ])
        return expanded_infostate

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
