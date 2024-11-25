from activation import activationFunctions
import numpy as np

seeds = [42, 22, 22]
seeds = [42, 22]

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

        initialized_layers: uma lista contendo duas listas, weights e functions
            - A lista de weights contendo as matrizes de pesos dos neurônios de cada camada
            - A lista de functions contém uma tupla com a função de ativação da camada e suas derivadas
    """

    # inicializa prevLayerNeurons com o número de neurônios da camada de input
    prevLayerNeurons = attrNum

    # Inicializa listas vazias de pesos e funções
    weights = []
    functions = []

    # Itera por todas as camadas
    for i, layer in enumerate(layers):

        np.random.seed(seeds[i])

        # Extrai número de neurônios e função de ativação da camada
        neuronNum, activation = layer

        # Gera a matriz de pesos da camada. Um row por neurônio, contendo um peso para cada neurônio da camada anterior + 1 para o bias
        layer_weights = np.random.randn(neuronNum, prevLayerNeurons + 1)

        # Guarda os valores obtidos nas respectivas listas
        weights.append(layer_weights)
        functions.append(activationFunctions[activation])

        # Atualiza prevLayerNeurons com o número de neurônios da camada atual
        prevLayerNeurons = neuronNum

    initialized_layers = [weights, functions]

    return initialized_layers
