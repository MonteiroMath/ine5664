from activation import activationFunctions
import numpy as np


'''
Valores válidos para funções de ativação:

SIGMOID
RELU
SOFTMAX

'''


def initLayers(layers, attrNum):
    """
    Função para inicializar as camadas da rede neural

    Recebe:

        Layers: array de tuplas. Cada tupla representa uma camada da rede neural e possui o formato (númeroDeNeurônios, idFunçãoDeAtivação)

        attrNum: número de atributos da camada de input

    Retorna:

        initialized_layers: uma lista contendo dicionários de parâmetros para cada camada
            - weights: matriz de pesos da camada
            - activation: função de ativação da camada
            - derivation: função de derivação da camada
    """

    # inicializa prevLayerNeurons com o número de neurônios da camada de input
    prevLayerNeurons = attrNum

    # inicializa lista de camadas vazia
    initialized_layers = []

    # Itera por todas as camadas
    for i, layer in enumerate(layers):

        # inicializa dicionário de parâmetros
        layerParams = {}

        # Extrai número de neurônios e função de ativação da camada
        neuronNum, activation = layer

        # Gera a matriz de pesos da camada. Um row por neurônio, contendo um peso para cada neurônio da camada anterior + 1 para o bias
        #layerParams["weights"] = np.random.randn(
        #    neuronNum, prevLayerNeurons + 1)
        layerParams["weights"] = np.random.uniform(low=-0.33, high=0.33,size=(neuronNum,prevLayerNeurons + 1))
        
        # Extrai funções de ativação e derivação da camada
        activationFunction, derivateFunction = activationFunctions[activation]
        layerParams["activation"] = activationFunction
        layerParams["derivation"] = derivateFunction

        # Atualiza prevLayerNeurons com o número de neurônios da camada atual
        prevLayerNeurons = neuronNum

        # Guarda os parâmetros da camada
        initialized_layers.append(layerParams)

    return initialized_layers
