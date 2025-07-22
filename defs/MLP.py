"""
Here we define a Multi-Layer Perceptron (MLP or DNN) using PyTorch.
This is incredibly crude and just a starting point.
"""

#region Imports

import json

import torch
import torch.nn as nn
import numpy as np

#endregion

#region MLP Class

"""
Here, we define a simple MLP class
"""

class MLP(nn.Module):

    def __init__(self, cfg_mlp):
        super(MLP, self).__init__()

        # Initialize MLP parameters
        self.i_dim = cfg_mlp['input_dim']
        self.o_dim = cfg_mlp['output_dim']
        self.hidden_units = cfg_mlp['hidden_units']
        self.activation = cfg_mlp.get('activation', nn.Tanh)()

        # -----Build MLP-----

        # Add input layer
        self.layers = [nn.Linear(in_features=self.i_dim, out_features=self.hidden_units[0]), self.activation]

        # Add hidden layers
        for i in range(len(self.hidden_units)-1):
            self.layers.append(nn.Linear(in_features=self.hidden_units[i], out_features=self.hidden_units[i+1]))
            self.layers.append(self.activation)

        # Add output (latent space) layer
        self.layers.append(nn.Linear(in_features=self.hidden_units[-1], out_features=self.latent_dim))
        self.layers.append(self.activation)

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):

        # Return the forward prop

        return self.mlp(input)
        