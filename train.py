from rna import rna
from cost import meanSquaredError, mseDerivative, binaryCrossEntropy, categoricalCrossEntropy
import numpy as np


def train(epochs, learningRate, startWeights, observations):

    costFunction = meanSquaredError
    costDerivative = mseDerivative
    weights = startWeights

    for n in range(epochs):

        predictions = []
        for observation in observations:

            attributes, label = observation
            attributes = np.array(attributes)
            prediction = rna(attributes, weights)
            predictions.append(prediction)
            cost = costFunction(prediction, label)

        predictions = np.array(predictions)

        # cost = costFunction(predictions, labels)

    
    return

np.random.seed(42)
layer_1_weights = np.random.randn(2, 3)
np.random.seed(22)
output_layer_weights = np.random.randn(3)

EPOCHS = 10
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