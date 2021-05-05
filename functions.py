import numpy as np


def relu(x: np.array):
    return np.maximum(0, x)


def d_relu(x: np.array):
    return 0 if x < 0 else 1


def sigmoid(x: np.array):
    return 1 / (1 + np.e ** (-x))


def d_sigmoid(x: np.array):
    return x * (1 - x)


def ms_error(predicted: np.array, real: np.array):
    return np.mean((predicted - real) ** 2)


def d_ms_error(predicted: np.array, real: np.array):
    return 2*(predicted - real)


def d_tanh(x: np.ndarray):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    dt = 1 - t ** 2
    return dt

