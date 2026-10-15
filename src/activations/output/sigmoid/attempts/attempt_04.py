import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sigmoid_z = sigmoid(z)
    return sigmoid_z * (1 - sigmoid_z)