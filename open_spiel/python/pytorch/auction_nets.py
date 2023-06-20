
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

INVALID_ACTION_PENALTY = -1e6

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

def string_to_activation(activation_string_or_func):
    activations = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }
    if callable(activation_string_or_func):
        return activation_string_or_func
    else:
        try:
            return activations[activation_string_or_func.lower()]
        except KeyError:
            raise ValueError(f'Invalid activation {activation_string_or_func}; valid activations are {list(activations.keys())}')

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std) # Somewhat arbitrary choice, but this is what CleanRL used
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def num_products_to_conv_layer(num_products):
    if num_products == 2:
        return nn.Conv2d
    elif num_products == 3:
        return nn.Conv3d
    else:
        # ConvND libraries exist online, could check those out
        raise ValueError(f"num_products must be 2 or 3, got {num_products}")

def build_conv_layer(num_products, in_channels, out_channels, kernel_size, std=1):
    conv_layer = num_products_to_conv_layer(num_products)
    # TODO: What if there is 1 product and 10 units and kernel size is 4? This says pad 1 on each side, but that's not enough, right?
    padding = (kernel_size - 1) // 2
    return layer_init(conv_layer(in_channels, out_channels, kernel_size, padding=padding), std=std)


class LinearWithMax(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearWithMax, self).__init__()

        self.linear = layer_init(nn.Linear(in_features * 2, out_features))

    def forward(self, x):
        """
        x: (batch_size, bundles, features)
        output: (batch_size, bundles, features)
        """

        # logging.info(x.shape)
        y = torch.cat([x, torch.max(x, dim=1, keepdim=True)[0].expand_as(x)], dim=2)
        return self.linear(y)

class ResidualLayer(nn.Module):
    def __init__(self, features, activation=nn.ReLU, add_skip_connection=True, add_max_features=True):
        super(ResidualLayer, self).__init__()

        self.add_skip_connection = add_skip_connection
        self.add_max_features = add_max_features

        if add_max_features:
            self.linear1 = LinearWithMax(features, features)
            self.linear2 = LinearWithMax(features, features)
        else:
            self.linear1 = layer_init(nn.Linear(features, features))
            self.linear2 = layer_init(nn.Linear(features, features))

        self.activation = string_to_activation(activation)()

    def forward(self, x):
        """
        x: (batch_size, bundles, features)
        output: (batch_size, bundles, features)
        """

        y = self.activation(self.linear1(x))
        y = self.linear2(y)
        return self.activation(x + y if self.add_skip_connection else y)

def build_auction_net_torso(observation_shape, hidden_sizes, activation=nn.ReLU, add_skip_connections=True, add_max_features=True):
    layers = []
    in_features, num_bundles = observation_shape
    num_hidden_features = hidden_sizes[0]

    if add_max_features:
        layers.append(LinearWithMax(in_features, num_hidden_features))
    else:
        layers.append(layer_init(nn.Linear(in_features, num_hidden_features), std=1))
    layers.append(string_to_activation(activation)())
    for i in range(len(hidden_sizes) - 1):
        layers.append(ResidualLayer(num_hidden_features, activation, add_skip_connections, add_max_features))
    return layers

class AuctionQNet(nn.Module):
    """
    For use in DQN. 

    TODO: avoid code duplication with AuctionNet?
    """
    def __init__(self, num_actions, observation_shape, device, hidden_sizes, activation=nn.ReLU, output_std=0.01, add_skip_connections=True, add_max_features=True):
        super(AuctionQNet, self).__init__()

        if not np.allclose(hidden_sizes, hidden_sizes[0]):
            raise ValueError("All hidden sizes must be equal")
        num_hidden_features = hidden_sizes[0]

        self.network = nn.Sequential(
            *build_auction_net_torso(observation_shape, hidden_sizes, activation, add_skip_connections, add_max_features), 
            layer_init(nn.Linear(num_hidden_features, 1), std=output_std)
        )
        
        self.device = device
        self.num_actions = num_actions
    
    def forward(self, x):
        """
        x: (batch_size, features, bundles)
        output: (batch_size, bundles)
        """
        x = x.permute(0, 2, 1) # (batch_size, bundles, features)
        return self.network(x).squeeze(-1)

class AuctionNet(nn.Module):
    """
    For use in PPO. Supplies both actor (action probabilities) and critic (state advantage) heads.
    """
    def __init__(self, num_actions, observation_shape, device, hidden_sizes, activation=nn.ReLU, actor_std=0.01, critic_std=1, add_skip_connections=True, add_max_features=True, use_torso=True):
        super(AuctionNet, self).__init__()

        if not np.allclose(hidden_sizes, hidden_sizes[0]):
            raise ValueError("All hidden sizes must be equal")
        num_hidden_features = hidden_sizes[0]
        in_features, num_bundles = observation_shape

        self.use_torso = use_torso

        if self.use_torso:
            self.torso = nn.Sequential(*build_auction_net_torso(observation_shape, hidden_sizes, activation, add_skip_connections, add_max_features))
            self.actor_head = layer_init(nn.Linear(num_hidden_features, 1), std=actor_std)
            self.critic_head = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(num_bundles*num_hidden_features, 1), std=critic_std),
            )
        else: # separate torso for each head
            self.actor_head = nn.Sequential(
                *build_auction_net_torso(observation_shape, hidden_sizes, activation, add_skip_connections, add_max_features), 
                layer_init(nn.Linear(num_hidden_features, 1), std=actor_std)
            )
            self.critic_head = nn.Sequential(
                *build_auction_net_torso(observation_shape, hidden_sizes, activation, add_skip_connections, add_max_features),
                nn.Flatten(),
                layer_init(nn.Linear(num_bundles*num_hidden_features, 1), std=critic_std),
            )

        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def forward(self, x):
        """
        x: (batch_size, features, bundles)
        output: (batch_size, bundles)
        """
        # print(x.shape)
        x = x.permute(0, 2, 1) # (batch_size, bundles, features)
        if self.use_torso:
            torso_output = self.torso(x)
        else:
            torso_output = x

        actor_output = self.actor_head(torso_output).squeeze(-1)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x4096 and 1728x1)
        critic_output = self.critic_head(torso_output)
        return actor_output, critic_output

    def actor(self, x):
        return self.forward(x)[0]

    def critic(self, x):
        return self.forward(x)[1]

    def actor_and_critic(self, x):
        return self.forward(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None: # TODO: Should you be taking this as a feature?
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits, critic_value = self.actor_and_critic(x)
        if torch.isnan(logits).any():
            raise ValueError("Training is messed up - logits are NaN")

        probs = CategoricalMasked(logits=logits, masks=legal_actions_mask, mask_value=self.mask_value)
        if action is None:
            action = probs.sample()

        # print("START")
        # print(x, logits, critic_value)
        # print("END")

        return action, probs.log_prob(action), probs.entropy(), critic_value, probs.probs

    def get_value(self, x):
        # logging.info(x.shape)
        return self.critic(x)


class ResidualConvLayer(nn.Module):
    def __init__(self, num_products, channels, kernel_size, activation=nn.ReLU, add_skip_connection=True):
        super(ResidualConvLayer, self).__init__()

        self.conv1 = build_conv_layer(num_products, channels, channels, kernel_size)
        self.conv2 = build_conv_layer(num_products, channels, channels, kernel_size)
        self.activation = string_to_activation(activation)()
        self.add_skip_connection = add_skip_connection

    def forward(self, x):
        """
        x: (batch_size, channels, *num_licenses)
        output: (batch_size, channels, *num_licenses)
        """
        y = self.activation(self.conv1(x))
        y = self.conv2(y)
        return self.activation(x + y if self.add_skip_connection else y)

class AuctionConvNet(nn.Module):
    def __init__(self, observation_shape, hidden_sizes, activation=nn.ReLU, kernel_size=3, actor_std=0.01, critic_std=1, add_skip_connections=True):
        super(AuctionConvNet, self).__init__()

        if not np.allclose(hidden_sizes, hidden_sizes[0]):
            raise ValueError("All hidden sizes must be equal")
        num_hidden_channels = hidden_sizes[0]

        layers = []
        in_channels = observation_shape[0]
        num_products = len(observation_shape[1:])

        layers.append(build_conv_layer(num_products, in_channels, num_hidden_channels, kernel_size))
        layers.append(string_to_activation(activation)())
        for i in range(len(hidden_sizes) - 1):
            layers.append(ResidualConvLayer(num_products, num_hidden_channels, kernel_size, activation, add_skip_connections))
        self.torso = nn.Sequential(*layers)

        self.actor_head = build_conv_layer(num_products, num_hidden_channels, 1, kernel_size, std=actor_std)
        self.critic_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(num_hidden_channels * np.prod(observation_shape[1:]), 1), std=critic_std),
        )

    def forward(self, x):
        torso_output = self.torso(x)
        actor_output = self.actor_head(torso_output)
        critic_output = self.critic_head(torso_output)
        return actor_output, critic_output

    def actor(self, x):
        return self.forward(x)[0]

    def critic(self, x):
        return self.forward(x)[1]