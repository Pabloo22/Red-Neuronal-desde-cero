import numpy as np


class LayerDense:

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.bias = 0.1 * np.random.randn(1, n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.dot(inputs, self.weights) + self.bias

    def __str__(self):
        return "W = " + str(self.weights) + "b = " + str(self.bias)
