from collections import OrderedDict
import torch.nn as nn

# The deep neural network
class DNN(nn.Module):

    def __init__(self, layers, dropout_rate, init_type):
        """
        Initializes the DNN.

        Args:
        layers: A list containing the number of neurons in each layer.
        dropout_rate: The dropout rate (probability of an element to be zeroed).
        init_type: The type of weight initialization ('xavier' or 'kaiming').
        """
        super(DNN, self).__init__()
        # Decide activation function based on the initialization type
        if init_type == 'xavier':
            self.activation = nn.Tanh()
        elif init_type == 'kaiming':
            self.activation = nn.LeakyReLU(negative_slope=0.01)  # Using Leaky ReLU for Kaiming initialization
        else:
            raise ValueError(f"Invalid init_type: {init_type}. Use 'kaiming' or 'xavier'.")
        
        self.layers = nn.Sequential(self._build_layers(layers, dropout_rate, init_type))

    def _build_layers(self, layers, dropout_rate, init_type):
        layer_list = []
        num_layers = len(layers)

        for i in range(num_layers - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            DNN._initialize_layer(linear_layer, init_type, zero_bias=(i < num_layers - 2))

            layer_list.append((f'layer_{i}', linear_layer))
            if i < num_layers - 2:
                layer_list.append((f'activation_{i}', self.activation))
                layer_list.append((f'dropout_{i}', nn.Dropout(dropout_rate)))

        return OrderedDict(layer_list)

    @staticmethod
    def _initialize_layer(layer, init_type, zero_bias=True):
        if init_type == 'kaiming':
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')  # Using Kaiming initialization for Leaky ReLU
        elif init_type == 'xavier':
            nn.init.xavier_uniform_(layer.weight)  # Using Xavier initialization for Tanh
        else:
            raise ValueError(f"Invalid init_type: {init_type}. Use 'kaiming' or 'xavier'.")

        if zero_bias:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)
