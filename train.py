import numpy as np
from rna import rna
from cost import costFunctions
from activation import activationFunctions
from initLayers import initLayers
from backpropagation import backpropagation


def train(epochs, learningRate, layers, observations):

    costFunction, costDerivative = costFunctions["MSE"]
    activationFunction, activationDerivative = activationFunctions["SIGMOID"]

    for n in range(epochs):

        # print(weights['output_layer_weights'])

        predictions = []
        for observation in observations:

            attributes, label = observation
            attributes = np.array(attributes)
            # prediction, combination, ol_input = rna(attributes, weights)
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


'''
np.random.seed(42)
layer_1_weights = np.random.randn(2, 3)
np.random.seed(500)
layer_2_weights = np.random.randn(2, 3)
np.random.seed(22)
output_layer_weights = np.random.randn(1, 3)
'''


EPOCHS = 1000
LEARNING_RATE = 0.1

'''
START_WEIGHTS = {
    'layer_1_weights': layers[0][0],
    'layer_2_weights': layers[1][0],
    'output_layer_weights': layers[2][0],

}
'''
observations = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


train(EPOCHS, LEARNING_RATE, layers, observations)
