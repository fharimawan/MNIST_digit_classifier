import numpy as np
from psutil._compat import xrange

class NeuralNetwork:
    def __init__(self, sizes):
        self.nlayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(size, x) for x, size in zip(size[:-1], sizes[1:])]

    

