from typing import Callable, List
from layer import LayerDense
import numpy as np


class NeuralNetwork:

    def __init__(self, topology: List[int],
                 act_func: Callable[[np.ndarray], np.ndarray],
                 d_act_func: Callable[[np.ndarray], np.ndarray]):
        self.layers = [LayerDense(topology[i], topology[i + 1]) for i in range(len(topology) - 1)]
        self.act_func = act_func
        self.d_act_func = d_act_func

    def forward_pass(self, inputs) -> List[np.ndarray]:

        out = [inputs]
        for L in range(len(self.layers)):
            z = self.layers[L].forward(out[-1])  # np.dot(inputs, self.weights) + self.bias
            a = self.act_func(z)
            out.append(a)

        return out

    def optimize(self, inputs: np.ndarray,
                 expected: np.ndarray,
                 d_cost_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 lr=0.05):

        prediction = self.forward_pass(inputs)
        deltas: List[np.ndarray or None] = []

        for L in reversed(range(len(self.layers))):
            a = prediction[L+1]

            # Backward pass
            if L == len(self.layers) - 1:
                # deltas[L] = d_cost_func(a, expected) * self.d_act_func(a)
                deltas.insert(0, d_cost_func(a, expected) * self.d_act_func(a))
            else:
                # deltas[L] = deltas[L+1] @ W_aux.T * self.d_act_func(a)
                deltas.insert(0, deltas[0] @ W_aux.T * self.d_act_func(a))

            # Gradient descent
            W_aux = self.layers[L].weights

            self.layers[L].bias -= np.mean(deltas[0], axis=0, keepdims=True) * lr
            self.layers[L].weights -= prediction[L].T @ deltas[0] * lr

    def train(self, outputs, l2_cost: Callable[[np.ndarray, np.ndarray], np.ndarray], steps=2500, lr=0.05):
        pass
