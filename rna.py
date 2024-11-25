import numpy as np
from activation import activationFunctions


def prepareInput(observation):
    """
    Inclui X0 = 1 no array de observações para multiplicar pelo BIAS.

    observation consiste em um array contendo os atributos de uma observação
    """
    return np.concatenate(([1], observation))


def forwardPass(input, weights, activationFunction):
    """
    Realizar um forward pass para uma camada

    Recebe:

        input: Array contendo as entradas da camada (atributos do input ou ativações da camada anterior)

        weights: matrix contendo os pesos de cada neurônio da camada

        activationFunction: função de ativação definida para uso na camada

    Retorna:

        activation: os valores calculados para a ativação do neurônio

        combination: os valores calculados na combinação linear do neurônio
    """

    combination = np.dot(input, weights.T)
    activation = activationFunction(combination)

    return activation, combination


def rna(input, layers):
    """
    Implementa a etapa de forward propagation da rede neural

    Recebe:

        input: Array contendo os atributos da observação para processamento

        layers: lista contendo dicionário com parâmetros de cada camada. Ver initLayers.

    """

    # inicializa prevActivations com os valores da camada de input
    prevActivations = input

    # iteração pelas camadas da rede neural
    for i in range(len(layers)):

        # extrai parâmetros da camada
        layerParams = layers[i]

        # Extrai os pesos da camada
        layerWeights = layerParams["weights"]

        # Extrai a função de ativação da camada
        activationFunction = layerParams["activation"]

        layerInput = prepareInput(prevActivations)

        # Realiza a forwardPass da camada
        activations, combinations = forwardPass(
            layerInput, layerWeights, activationFunction)

        # Guarda os valores intermediários produzidos
        layerParams["intermediate"] = (layerInput, combinations)

        # atualiza prevActivations para os valores de ativação produzidos nessa camada
        prevActivations = activations

    return activations
