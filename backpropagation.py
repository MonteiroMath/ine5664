import numpy as np


def backpropagation(layers, intermediateValues, costD, learningRate):

    nextLaywerWeights = None
    nextLayerErrorSignals = None

    weights, functions = layers

    reversedWeights = list(reversed(weights))
    reversedFunctions = list(reversed(functions))
    reversedIntermediate = list(reversed(intermediateValues))
    adjustedWeights = []

    for i in range(len(weights)):

        laywerWeights = reversedWeights[i]
        activationFunction, activationDerivative = reversedFunctions[i]

        newWeights, errorsSignals = backpropagateLayer(
            (laywerWeights, nextLaywerWeights),
            reversedIntermediate[i],
            learningRate,
            costD,
            activationDerivative,
            nextLayerErrorSignals
        )

        nextLaywerWeights = laywerWeights
        nextLayerErrorSignals = errorsSignals
        adjustedWeights.insert(0, newWeights)

    return adjustedWeights


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

    newWeights = currentWeights - gradients
    return newWeights, errorSignals
