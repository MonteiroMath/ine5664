import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def identity(x):
    return x


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    # usar para classificação multi-classes

    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
