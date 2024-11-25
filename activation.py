import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDerivative(x):
    sigmoidResult = sigmoid(x)
    return sigmoidResult * (1 - sigmoidResult)


def identity(x):
    return x


def ReLU(x):
    return np.maximum(0, x)

def ReluDerivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # usar para classificação multi-classes

    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmaxDerivative(x):
    return


activationFunctions = {
    "SIGMOID": (sigmoid, sigmoidDerivative),
    "RELU": (ReLU, ReluDerivative),
    "SOFTMAX": (softmax, softmaxDerivative)
}