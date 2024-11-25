from activation import activationFunctions
import numpy as np

seeds = [42, 22, 22]


seeds = [42, 22]


def initLayers(layers, attrNum):

    prevLayerNeurons = attrNum

    weights = []
    functions = []

    for i, layer in enumerate(layers):

        np.random.seed(seeds[i])

        neuronNum, activation = layer
        layer_weights = np.random.randn(neuronNum, prevLayerNeurons + 1)

        weights.append(layer_weights)
        functions.append(activationFunctions[activation])

        prevLayerNeurons = neuronNum

    initialized_layers = [weights, functions]

    return initialized_layers
