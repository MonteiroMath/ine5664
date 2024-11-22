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
from activation import sigmoid


def prepareInput(observation):
    # Inclui X0 = 1 no array de observações para multiplicar pelo BIAS.
    return np.concatenate(([1], observation))


def forwardPass(input, weights, activationFunction):

    adjustedInput = prepareInput(input)

    combination = np.dot(adjustedInput, weights.T)
    activation = activationFunction(combination)

    return activation, combination, adjustedInput


def rna(input, weights):

    # extração dos pesos de cada camada
    layer_1_weights = weights['layer_1_weights']
    layer_2_weights = weights['layer_2_weights']
    output_layer_weights = weights['output_layer_weights']
    activationFunction = sigmoid

    intermediateValues = {}

    # Operações da camada 1
    layer_1_activations, layer_1_combinations, layer_1_input = forwardPass(
        input, layer_1_weights, activationFunction)

    intermediateValues["layer_1"] = (
        layer_1_input, layer_1_combinations)

    '''
    # Operações da camada 2
    layer_2_activations, layer_2_combinations = forwardPass(layer_1_activations, layer_2_weights, activationFunction)

    # Operações da camada de output
    output_activations, output_combinations = forwardPass(layer_2_activations, output_layer_weights, activationFunction)
    '''

    # Operações da camada de output
    output_layer_activation, output_layer_combination, output_layer_input = forwardPass(
        layer_1_activations, output_layer_weights, activationFunction)
    intermediateValues["output_layer"] = (
        output_layer_input, output_layer_combination)

    return output_layer_activation, intermediateValues


'''
#! Rede com 2 hidden layers de 3 neurons cada

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

'''
#! Rede com 1 hidden layer de 2 neurons
# Inicialização dos pesos
np.random.seed(42)

layer_1_weights = np.random.randn(2, 3)
np.random.seed(32)
#layer_2_weights = np.random.randn(3, 4)
np.random.seed(22)
output_layer_weights = np.random.randn(3)
weights = {
    'layer_1_weights': layer_1_weights,
    'layer_2_weights': None,
    'output_layer_weights': output_layer_weights,
}

results = rna([0, 0], weights)
print(results)
'''
