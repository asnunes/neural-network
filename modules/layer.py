from modules.mathwrapper import RandomInitializer as rinit
from modules.mathwrapper import Sigmoid as sm
from modules.mathwrapper import Operation as op


class Layer:
    def __init__(self, num_of_units, number_of_inputs):
        self.num_of_units = num_of_units
        self.weight = rinit.generate(number_of_inputs, num_of_units)

        self.output = None

    def process(self, x):
        z = op.vec_prod(x, self.weight)
        self.output = sm.sigmoid(z)
