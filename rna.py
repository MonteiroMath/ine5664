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


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def limiar(x):
    return 1 if x >= 0 else 0


X0 = 1
WEIGHTS = [0, 0, 0]
LEARNING_RATE = 0.5

weightArray = np.array(WEIGHTS)

observations = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

labels = [0, 1, 1, 1]

for n in range(10):
    for i, observation in enumerate(observations):

        inputArray = np.array([X0] + observation)
        neuron_output = np.dot(inputArray, weightArray)
        # activation_output = sigmoid(neuron_output)
        prediction = limiar(neuron_output)

        if prediction != labels[i]:

            weightArray = weightArray + LEARNING_RATE * \
                inputArray * (labels[i] - prediction)

    print(weightArray)
