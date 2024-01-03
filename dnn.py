"""
PINN training for Boussinesq 2d
@author: Reza Salatin
@email: reza.salatin@whoi.edu
December 2023
w/ Pytorch
"""

from collections import OrderedDict
import torch.nn as nn

# The deep neural network
class DNN(nn.Module):

    def __init__(self, layers):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(self._build_layers(layers))

    def _build_layers(self, layers):
        layer_list = []
        num_layers = len(layers)

        for i in range(num_layers - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            DNN._initialize_layer(linear_layer, zero_bias=(i < num_layers - 2))

            layer_list.append((f'layer_{i}', linear_layer))
            if i < num_layers - 2:
                layer_list.append((f'activation_{i}', nn.Tanh()))

        return OrderedDict(layer_list)

    @staticmethod
    def _initialize_layer(layer, zero_bias=True):
        nn.init.xavier_uniform_(layer.weight)
        if zero_bias:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


# the deep neural network
class DNN_archive(nn.Module):
    
    def __init__(self, layers):
        super(DNN_archive, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            linear_layer = nn.Linear(layers[i], layers[i+1])

            # Xavier initialization
            nn.init.xavier_uniform_(linear_layer.weight)
            # Set the biases to zero
            nn.init.zeros_(linear_layer.bias)

            layer_list.append(('layer_%d' % i, linear_layer))
            layer_list.append(('activation_%d' % i, self.activation()))

        # Apply Xavier initialization to the last layer as well
        last_linear_layer = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(last_linear_layer.weight)

        # Initialize the biases of the last layer to zero
        nn.init.zeros_(last_linear_layer.bias)

        layer_list.append(('layer_%d' % (self.depth - 1), last_linear_layer))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = nn.Sequential(layerDict)
        
    def forward(self, x):
        return self.layers(x)