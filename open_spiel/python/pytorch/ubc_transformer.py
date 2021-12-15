import torch
from torch import nn
import math

class AuctionTransformer(nn.Module):
    """
    A transformer model designed for our clock auction games.
    """
    def __init__(self, num_players, num_products, input_size, output_size, d_model=32, nhead=1, feedforward_dim=128, dropout=0.1, num_layers=1):        
        """
        Initialize the model.

        Args:
        - num_players: number of bidders in the auction
        - num_products: number of products being bid on in the auction
        - output_size: number of possible actions that bidders can take (likely \prod_i(1+supply_i) )
        - d_model: dimension of hidden state
        - num_layers: number of recurrent layers to stack in model
        - rnn_model: type of torch model to use; options are "lstm" and "rnn"
        """
        super(AuctionTransformer, self).__init__()

        # TODO: don't hardcode 
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

        self.embedding = nn.Linear(input_size_per_round, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, feedforward_dim, dropout, batch_first=True)
        # TODO: layer norm?
        # encoder_norm = nn.LayerNorm(self.dmodel)
        # and pass norm=encoder_norm into encoder
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        Apply the model to a batch of infostate tensors.

        Arguments:
        - x: tuple of
            - (batch_size, num_rounds, features) tensor of infostates
            - number of unmasked rounds in each example

        Outputs:
        - tensor of logits or Q-values for each action
        """


        (infostate_tensors, sequence_lengths) = x
        (batch_size, num_rounds, _) = infostate_tensors.shape

        mask = torch.zeros(batch_size, num_rounds)
        for example in range(batch_size):
            mask[example, sequence_lengths[example]:] = float('-inf')

        out = self.embedding(infostate_tensors)            # (batch_size, num_rounds, d_model)
        out = self.pos_encoder(out)                        # (batch_size, num_rounds, d_model)
        out = self.encoder(out, src_key_padding_mask=mask) # (batch_size, num_rounds, d_model)
        out = out[:, -1, :]                                # (batch_size, d_model)
        out = self.output_layer(out)                       # (batch_size, output_size)

        return out

    def prep_batch(self, infostate_list):
        """
        Prepare a list of infostate tensors to be used as an input to the network.

        Args:
        - infostate_list: a list of infostate tensors
        
        Returns: (batch_size, num_rounds, num_features) tensor with batch of infostates 
        """

        sequence_lengths = [len(t) for t in infostate_list]
        batch_size = len(infostate_list)
        max_length = max(sequence_lengths)
        num_features = infostate_list[0].shape[1]

        batch_tensor = torch.zeros(batch_size, max_length, num_features)
        for i, t in enumerate(infostate_list):
            batch_tensor[i, :sequence_lengths[i], :] = t[:, :]

        return (batch_tensor, sequence_lengths)

    def reshape_infostate(self, infostate_tensor):
        """
        Expand a flat infostate tensor into a per-round tensor

        Args:
        - infostate_tensor: flat list describing game infostate
    
        Outputs: tensor of shape (num rounds, model input size)
        
        TODO: copied from RNN class. Refactor...
        """

        prefix_len = 3 * self.num_players + 1 + self.num_products
        suffix_len_per_round = 4 * self.num_products
        
        # split tensor into (per-auction, per-round) features
        prefix = torch.tensor(infostate_tensor[:prefix_len])
        suffix = torch.tensor(infostate_tensor[prefix_len:])
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

class PositionalEncoding(nn.Module):
    def __init__(self, dimensions, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dimensions)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0) / dimensions))
        pe[:, 0::2] = torch.sin(position * div_term) # All even indices of the embedding dim.
        pe[:, 1::2] = torch.cos(position * div_term) # All odd indices of the embedding dim.
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Note: most NLP models _add_ positional encoding
        # Concatenate instead because we don't learn an embedding...
        # x = torch.hstack([x, self.pe[:x.size(0), :]])
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
