import numpy as np


class RandomInitializer:
    @staticmethod
    def generate(input_dim, num_of_units):
        return np.random.rand(input_dim, num_of_units)


class Dimensions:
    @staticmethod
    def size_x(mtx):
        return mtx.shape[0]

    @staticmethod
    def size_y(mtx):
        return mtx.shape[1]

    @staticmethod
    def size(mtx):
        return mtx.shape


class Sigmoid:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

class Operation:
    @staticmethod
    def vec_prod(x, y):
        return np.dot(x, y)
