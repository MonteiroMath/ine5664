import numpy as np


def backpropagation(layers, intermediateValues, costD, learningRate):

    reversedIntermediate = list(reversed(intermediateValues))
    nextLaywerWeights = None
    nextLayerErrorSignals = None

    for i, layer in enumerate(reversed(layers)):

        weights, (activationFunction, activationDerivative) = layer

        errorsSignals = backpropagateLayer(
            (weights, nextLaywerWeights),
            reversedIntermediate[i],
            learningRate,
            costD,
            activationDerivative,
            nextLayerErrorSignals
        )

        nextLaywerWeights = weights
        nextLayerErrorSignals = errorsSignals


def backpropagateLayer(weights, intermediateValues, learningRate, costD, activationDerivative, nextLayerErrorSignals=None):

    currentWeights, nextLayerWeights = weights
    layerInput, combinations = intermediateValues

    activationD = activationDerivative(combinations)

    if nextLayerErrorSignals is None:
        # camada de output
        errorSignals = costD * activationD
    else:
        # camadas ocultas
        propagatedErrorSignals = np.dot(
            nextLayerErrorSignals, nextLayerWeights[:, 1:])
        errorSignals = propagatedErrorSignals * activationD

    gradients = np.outer(errorSignals, layerInput) * learningRate

    currentWeights -= gradients
    return errorSignals
