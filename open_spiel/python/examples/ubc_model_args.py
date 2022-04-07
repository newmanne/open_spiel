from open_spiel.python.pytorch import ubc_nfsp, ubc_dqn, ubc_rnn, ubc_transformer, ubc_flat_mlp
from open_spiel.python.examples.ubc_utils import handcrafted_size

def lookup_model_and_args(model_name, state_size, num_actions, num_players, num_types, num_products=None):
    """
    lookup table from (model name) to (function, default args)
    """
    if model_name != 'mlp' and num_products is None:
        raise ValueError("Num products can only be None if model is MLP")

    if model_name == 'mlp': 
        model_class = ubc_dqn.MLP
        # TODO: If you wanted the MLP to do somethign different (e.g., only get the current observation, or the past N observations, you would need to make some changes to the args here)
        default_model_args = {
            'input_size': handcrafted_size(num_actions, num_products),
            'hidden_sizes': [128],
            'output_size': num_actions,
            'num_players': num_players,
            'num_products': num_products, 
        }
    elif model_name == 'recurrent':
        model_class = ubc_rnn.AuctionRNN
        default_model_args = {
            'num_players': num_players,
            'num_products': num_products, 
            'num_types': num_types, 
            'input_size': state_size, 
            'hidden_size': 128,
            'output_size': num_actions,
            'nonlinearity': 'tanh',
        }
    elif model_name == 'flatmlp':
        model_class = ubc_flat_mlp.FlatMLP
        default_model_args = {
            'num_players': num_players,
            'num_products': num_products, 
            'num_types': num_types, 
            'input_size': state_size, # TODO: This isn't correct, b/c you to multiply by whatever MAX_ROUNDS you are using here
            'hidden_sizes': [128],
            'output_size': num_actions,
        }
    elif model_name == 'transformer': 
        model_class = ubc_transformer.AuctionTransformer
        default_model_args = {
            'num_players': num_players,
            'num_products': num_products,
            'input_size': state_size, 
            'output_size': num_actions, 
            'd_model': 32,
            'nhead': 1, 
            'feedforward_dim': 128, 
            'dropout': 0.1, 
            'num_layers': 1,
        }
    else: 
        raise ValueError(f'Unrecognized model {model_name}')
    
    return model_class, default_model_args
