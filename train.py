import numpy as np
from rna import rna
from cost import costFunctions
from initLayers import initLayers
from backpropagation import backpropagation


def train(epochs, learningRate, layers, observations, costF):

    costFunction, costDerivative = costFunctions[costF]

    for n in range(epochs):

        predictions = []
        for observation in observations:

            attributes, label = observation
            attributes = np.array(attributes)
            prediction, intermediateValues = rna(attributes, layers)
            predictions.append(prediction)

            # invoca a função de custo
            cost = costFunction(prediction, label)
            print(prediction)

            # Calcula a derivada do custo para a observação corrente
            costD = costDerivative(prediction, label)

            adjustedWeights = backpropagation(
                layers, intermediateValues, costD, learningRate)
            layers[0] = adjustedWeights