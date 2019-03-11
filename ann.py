"""
Created at 11/03/19 by Alexandre Nunes - MIT License
"""

from modules.layer import Layer
from modules.mathwrapper import Dimensions as dim
import numpy as np


class NeuralNetwork:
    """
     layers_info is a array which dimension is the number of layers and each
     value is the number of layer`s units.
    """
    def __init__(self, x, layers_info, y):
        self.x = x
        self.layers = self._init_layers(layers_info)
        self.y = y

        self.output = None
        self.error = None

    def feed_forward(self):
        inputs = [self.x]

        for layer in self.layers:
            layer.process(inputs[-1])
            inputs.append(layer.output)

        self.output = inputs[-1]
        return self.output

    def _init_layers(self, layers_info):
        if not layers_info:
            raise IOError('Neural Network must have at least one layer...')

        layers = [self._generate_first_layer(layers_info[0])]

        for num_of_units in layers_info:
            layers.append(Layer(num_of_units, layers[-1].num_of_units))

        return layers

    def _generate_first_layer(self, num_of_units):
        return Layer(num_of_units, dim.size_y(self.x))


x = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0],[1],[1],[0]), dtype=float)
layers_info = [4, 1]

nn = NeuralNetwork(x, layers_info, y)
print(nn.feed_forward())
