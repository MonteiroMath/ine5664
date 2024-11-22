from rna import rna
from cost import costFunctions
from activation import activationFunctions
import numpy as np


def backpropagation(weights, intermediateValues, learningRate, costD, activationDerivative, nextLayerErrorSignals=None):

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


def train(epochs, learningRate, startWeights, observations):

    costFunction, costDerivative = costFunctions["MSE"]
    activationFunction, activationDerivative = activationFunctions["SIGMOID"]

    weights = startWeights

    for n in range(epochs):

        # print(weights['output_layer_weights'])

        predictions = []
        for observation in observations:

            attributes, label = observation
            attributes = np.array(attributes)
            # prediction, combination, ol_input = rna(attributes, weights)
            prediction, intermediateValues = rna(attributes, weights)
            predictions.append(prediction)

            # invoca a função de custo
            cost = costFunction(prediction, label)
            print(cost)

            # Calcula a derivada do custo para a observação corrente
            costD = costDerivative(prediction, label)

            # Backpropagation para camada de output

            errorSignals = backpropagation((weights["output_layer_weights"], None),
                                           intermediateValues["output_layer"],
                                           learningRate,
                                           costD,
                                           activationDerivative,
                                           None
                                           )

            # Backpropagation para a layer 1

            propagationWeights = (weights["layer_1_weights"],
                                  weights["output_layer_weights"])

            errorSignals = backpropagation(propagationWeights,
                                           intermediateValues["layer_1"],
                                           learningRate,
                                           costD,
                                           activationDerivative,
                                           errorSignals
                                           )


# predictions = np.array(predictions)
# cost = costFunction(predictions, labels)
np.random.seed(42)
layer_1_weights = np.random.randn(2, 3)
np.random.seed(500)
layer_2_weights = np.random.randn(2, 3)
np.random.seed(22)
output_layer_weights = np.random.randn(1, 3)

EPOCHS = 1000
LEARNING_RATE = 0.1
START_WEIGHTS = {
    'layer_1_weights': layer_1_weights,
    'layer_2_weights': layer_2_weights,
    'output_layer_weights': output_layer_weights,


}

observations = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


train(EPOCHS, LEARNING_RATE, START_WEIGHTS, observations)
