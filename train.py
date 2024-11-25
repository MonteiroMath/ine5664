import numpy as np
from rna import rna
from cost import costFunctions
from initLayers import initLayers
from backpropagation import backpropagation


def train(epochs, learningRate, layers, observations):

    costFunction, costDerivative = costFunctions["MSE"]

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

            backpropagation(layers, intermediateValues, costD, learningRate)


# Lista no formato [(neurons, activation), (neurons, activation), (neurons, activation)]
# Cada tupla representa uma camada
layers = [
    (2, 'SIGMOID'),
    # (2, 'SIGMOID'),
    (1, 'SIGMOID')
]

layers = initLayers(layers, 2)

EPOCHS = 1000
LEARNING_RATE = 0.1

observations = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


train(EPOCHS, LEARNING_RATE, layers, observations)
