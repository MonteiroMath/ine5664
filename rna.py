import numpy as np
from activation import activationFunctions


def prepareInput(observation):
    # Inclui X0 = 1 no array de observações para multiplicar pelo BIAS.
    return np.concatenate(([1], observation))


def forwardPass(input, weights, activationFunction):

    adjustedInput = prepareInput(input)

    combination = np.dot(adjustedInput, weights.T)
    activation = activationFunction(combination)

    return activation, combination, adjustedInput


def rna(input, layers):

    intermediateValues = []

    prevActivations = input

    for layer in layers:

        weights, (activationFunction, activationDerivative) = layer

        activations, combinations, layerInput = forwardPass(
            prevActivations, weights, activationFunction)

        intermediateValues.append((layerInput, combinations))
        prevActivations = activations

    return activations, intermediateValues
