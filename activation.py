import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def limiar(x):
    return 1 if x >= 0 else 0