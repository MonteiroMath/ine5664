'''
Steps:

- Define representation of neurons
    - Just mathematical operations
- Define representation of layers
    - Each layers will be represented by an array of arrays (or matrix). Each array inside it represents the weights of one neuron
    - Ex: [[w11, w12, w13], [w21,w22,23], [w31,w32,w33] ]
- Define how to connect neurons
    - Outputs of the previous layers goes to an array that become the input array for the next layer
- Define number of neurons and layers
    - via parameters
- Define activation functions
    - via parameters
- Define loss functions
    - via parameters
- Define backward propagation method
    - #todo
- Define training function and datasets for use
    - #todo

- Try implementing a simple network with a fixed number of layers, number of neurons and activation functions and see
what we get
'''
import numpy as np
from activation.activation import limiar, sigmoid


def prepareInputArray(observation):
    # Inclui X0 = 1 no array de observações para multiplicar pelo BIAS.
    return np.array([1] + observation)


def rna(input, weights):

    #extração dos pesos de cada camada
    layer_1_weights = weights['layer_1_weights']
    layer_2_weights = weights['layer_2_weights']
    output_layer_weights = weights['output_layer_weights']


    layer_1_input = prepareInputArray(input)
    layer_1_outputs = []

    for neuron_weights in layer_1_weights:
        output = np.dot(layer_1_input, neuron_weights)
        activation = sigmoid(output)
        layer_1_outputs.append(activation)

    layer_2_input = prepareInputArray(layer_1_outputs)

    layer_2_outputs = []

    for neuron_weights in layer_2_weights:
        output = np.dot(layer_2_input, neuron_weights)
        activation = sigmoid(output)
        layer_2_outputs.append(activation)

    output_layer_input = prepareInputArray(layer_2_outputs)

    output = np.dot(output_layer_input, output_layer_weights)
    activation = sigmoid(output)
    return activation


# Inicialização dos pesos
np.random.seed(42)
# creates a 3x3 matrix: 3 neurons with 3 weights each
layer_1_weights = np.random.randn(3, 3)
np.random.seed(32)
layer_2_weights = np.random.randn(3, 4)
np.random.seed(22)
output_layer_weights = np.random.randn(4)
weights = {
    'layer_1_weights': layer_1_weights,
    'layer_2_weights': layer_2_weights,
    'output_layer_weights': output_layer_weights,
}

results = rna([0, 0], weights)
print(results)

'''
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

    return weightArray
'''
