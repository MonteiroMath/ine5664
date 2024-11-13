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
    return 1/(1 + np.exp(np.dot(-1, x)))

VIES = 3
W0 = 1

inputArray = np.array([W0])
weightArray = np.array([VIES])

neuron_output = np.dot(inputArray, weightArray)

activation_output = sigmoid(neuron_output)

print(activation_output)





