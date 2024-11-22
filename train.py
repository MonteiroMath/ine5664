from rna import rna
from cost import meanSquaredError, mseDerivative, binaryCrossEntropy, categoricalCrossEntropy
from activation import sigmoidDerivative
import numpy as np


def train(epochs, learningRate, startWeights, observations):

    costFunction = meanSquaredError
    costDerivative = mseDerivative
    activationDerivative = sigmoidDerivative
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

            weights['output_layer_weights'] -= gradient

            # Backpropagation para a layer 1

            prevErrorSignal = errorSignal

            layerInput, combination = intermediateValues["layer_1"]
            activationD = activationDerivative(combination)
            errorSignals = prevErrorSignal * \
                weights["output_layer_weights"][:, 1:] @ activationD
            gradients = np.outer(errorSignals, layerInput) * learningRate
            weights["layer_1_weights"] -= gradients


# predictions = np.array(predictions)
# cost = costFunction(predictions, labels)
np.random.seed(42)
layer_1_weights = np.random.randn(2, 3)
np.random.seed(22)
output_layer_weights = np.random.randn(1, 3)

EPOCHS = 1000
LEARNING_RATE = 0.1
START_WEIGHTS = {
    'layer_1_weights': layer_1_weights,
    'layer_2_weights': None,
    'output_layer_weights': output_layer_weights,
}

observations = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


train(EPOCHS, LEARNING_RATE, START_WEIGHTS, observations)
