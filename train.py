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
            layerInput, combination, activation = intermediateValues["output_layer"]

            activationD = activationDerivative(combination)

            errorSignal = costD * activationD

            #! layerInput[0] é sempre 1, assegurando o ajuste correto para o bias
            gradient = errorSignal * layerInput * learningRate
            weights['output_layer_weights'] -= gradient

            # Backpropagation para a layer 1

            prevActivationD = activationD
            layerInput, combination, activation = intermediateValues["layer_1"]

            # para neuron 1
            activationD = activationDerivative(combination[0])
            errorSignal = costD * prevActivationD * \
                weights["output_layer_weights"][1] * activationD
            gradient = errorSignal * layerInput * learningRate
            weights["layer_1_weights"][0] -= gradient

            # para neuron 2
            activationD = activationDerivative(combination[1])
            errorSignal = costD * prevActivationD * \
                weights["output_layer_weights"][2] * activationD

            gradient = errorSignal * layerInput * learningRate
            weights["layer_1_weights"][1] -= gradient


# predictions = np.array(predictions)
# cost = costFunction(predictions, labels)
np.random.seed(42)
layer_1_weights = np.random.randn(2, 3)
np.random.seed(22)
output_layer_weights = np.random.randn(3)

EPOCHS = 100
LEARNING_RATE = 0.5
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
