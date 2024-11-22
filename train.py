from rna import rna
from cost import costFunctions
from activation import activationFunctions
import numpy as np


def hiddenBackpropagation(weights, intermediateValues, prevErrorSignal, learningRate, activationDerivative):

    currentWeights, nextLayerWeights = weights
    layerInput, combination = intermediateValues

    activationD = activationDerivative(combination)

    # errorSignals = prevErrorSignal * \
    #    nextLayerWeights[:, 1:] * activationD

    propagatedErrorSignals = np.dot(prevErrorSignal, nextLayerWeights[:, 1:])

    errorSignals = propagatedErrorSignals * activationD

    gradients = np.outer(errorSignals, layerInput) * learningRate

    newWeights = currentWeights - gradients
    return (newWeights, errorSignals)


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
            layerInput, combination = intermediateValues["output_layer"]

            activationD = activationDerivative(combination)
            errorSignal = costD * activationD

            #! layerInput[0] é sempre 1, assegurando o ajuste correto para o bias
            gradient = errorSignal * layerInput * learningRate
            # assegura o formato correto do gradiente para a operação de subtração abaixo
            gradient = gradient.reshape(1, -1)

            weights['output_layer_weights'] -= gradient

            # Backpropagation para a layer 1

            propagationWeights = (weights["layer_1_weights"],
                                  weights["output_layer_weights"])

            newWeights, errorSignal = hiddenBackpropagation(propagationWeights,
                                                            intermediateValues["layer_1"],
                                                            errorSignal,
                                                            learningRate,
                                                            activationDerivative)

            weights["layer_1_weights"] = newWeights

            '''
            # Backpropagation para a layer 2
            propagationWeights = (weights["layer_2_weights"],
                                  weights["layer_1_weights"])

            newWeights, errorSignal = hiddenBackpropagation(propagationWeights,
                                                            intermediateValues["layer_2"],
                                                            errorSignal,
                                                            learningRate,
                                                            activationDerivative)

            weights["layer_2_weights"] = newWeights
            '''

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
