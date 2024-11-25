from activation import activationFunctions
import numpy as np

seeds = [42, 22, 22]


def initLayers(layers, attrNum):
    

    prevLayerNeurons = attrNum
    initialized_layers = []

    for i, layer in enumerate(layers):

        np.random.seed(seeds[i])

        neuronNum, activation = layer
        layer_weights = np.random.randn(neuronNum, prevLayerNeurons + 1)

        initialized_layers.append(
            (layer_weights, activationFunctions[activation]))
        
        prevLayerNeurons = neuronNum

    return initialized_layers
