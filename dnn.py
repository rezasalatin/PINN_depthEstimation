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
