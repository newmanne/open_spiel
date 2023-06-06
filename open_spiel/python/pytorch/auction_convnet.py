
import numpy as np
import torch
import torch.nn as nn

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

class ResidualLayer(nn.Module):
    def __init__(self, num_products, channels, kernel_size, activation=nn.ReLU, add_skip_connection=True):
        super(ResidualLayer, self).__init__()

        self.conv1 = build_conv_layer(num_products, channels, channels, kernel_size)
        self.conv2 = build_conv_layer(num_products, channels, channels, kernel_size)
        self.activation = activation()
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
        layers.append(activation())
        for _ in range(len(hidden_sizes) - 1):
            layers.append(ResidualLayer(num_products, num_hidden_channels, kernel_size, activation, add_skip_connections))
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
