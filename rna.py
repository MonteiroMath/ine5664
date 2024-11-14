'''
Steps:

- Define representation of neurons
- Define representation of layers
- Define how to connect neurons
- Define number of neurons and layers
- Define activation functions
- Define loss functions
- Define backward propagation method
- Define training function and datasets for use

- Try implementing a simple network with a sigmoid function and see where we get
'''
import numpy as np
from activation.activation import limiar


def prepareInputArray(observation):
    # Inclui X0 = 1 no array de observações para multiplicar pelo BIAS.
    return np.array([1] + observation)


def train(epochs, learningRate, weights, observations, labels):

    weightArray = np.array(weights)

    for n in range(epochs):
        for i, observation in enumerate(observations):

            inputArray = prepareInputArray(observation)
            # produto entre input e pesos
            neuron_output = np.dot(inputArray, weightArray)
            # activation_output = sigmoid(neuron_output)
            prediction = limiar(neuron_output)

            if prediction != labels[i]:

                # Ajusta pesos em caso de erro
                weightArray = weightArray + learningRate * \
                    inputArray * (labels[i] - prediction)

        print(weightArray)
