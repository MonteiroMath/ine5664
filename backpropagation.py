import numpy as np


def backpropagation(layers, intermediateValues, costD, learningRate):

    """
    
    Implementa a operação da retropropagação para toda a rede

    Recebe: 

        layers: lista contendo pesos e funções de ativação/derivadas. Ver initLayers para mais detalhes

        intermediateValues: lista contendo os valores intermediários produzidos na etapa de forward propagation

        costD: derivada da função de custo, já calculada

        learningRate: parâmetro de learningRate definido inicialmente
    
    Retorna:

        - Lista contendo os pesos ajustados após a retropropagação
    """

    # inicializa nextLayerWeights e nextLayerErrorSignals com None (Começa pela última camada)
    nextLayerWeights = None
    nextLayerErrorSignals = None

    # Extrai valores de pesos e funções
    weights, functions = layers

    # Reverte todas as litas para iterar a partir da última camada
    reversedWeights = list(reversed(weights))
    reversedFunctions = list(reversed(functions))
    reversedIntermediate = list(reversed(intermediateValues))

    # Inicializa uma lista de pesos ajustados
    adjustedWeights = []

    # itera pelas camadas (começando pela última e indo até a primeira)
    for i in range(len(weights)):

        # Obtém pesos da camada
        layerWeights = reversedWeights[i]

        # Obtém função para calcular derivada da função de ativação da camada
        activationFunction, activationDerivative = reversedFunctions[i]

        # Obtém pesos ajustados e sinais de erro da camada atual
        newWeights, errorsSignals = backpropagateLayer(
            (layerWeights, nextLayerWeights),
            reversedIntermediate[i],
            learningRate,
            costD,
            activationDerivative,
            nextLayerErrorSignals
        )

        # Atualiza nextLayerWeights e nextLayerErrorSignals com os valores correspondentes da camada atual
        nextLayerWeights = layerWeights
        nextLayerErrorSignals = errorsSignals

        # Insere os valores dos pesos ajustados na lista de pessoas ajustados, assegurando a ordem correta
        adjustedWeights.insert(0, newWeights)

    return adjustedWeights


def backpropagateLayer(weights, intermediateValues, learningRate, costD, activationDerivative, nextLayerErrorSignals=None):


    """ 
    
    Realiza a operação de backpropagação em uma camada
    
    Recebe:

        weights: tupla contendo os pesos da layer atual e os pesos da layer seguinte

        intermediateValues: tupla contendo os valores intermediários relevantes para a camada (inputs recebidos e combinações produzidas)

        learningRate: parâmetro de learningRate

        costD: valor calculado para a derivada da função de custo

        activationDerivative: função para cálculo da derivada da função de ativação

        nextLayerErrorSignals: Error signals da camada seguinte

    Retorna:

        newWeights: os pesos ajustados após o processo de retropropagação

        errorSignals: os sinais de erro produzidos na camada, que serão propagados para a camada anterior na próxima etapa
    """

    # Extrai pesos da camada atual e da camada seguinte
    currentWeights, nextLayerWeights = weights

    # extrai inputs recebidos pela camada atual e combinações lineares que produziu
    layerInput, combinations = intermediateValues

    # calcula a derivada da função de ativação da camada
    activationD = activationDerivative(combinations)

    # Se nextLayerErrorSignals é None, calcula o errorSignals para a camada de output. Do contrário, propaga o error signal para as camadas anteriores
    if nextLayerErrorSignals is None:
        # camada de output
        errorSignals = costD * activationD
    else:
        # camadas ocultas

        # propaga o ErrorSignal da camada seguinte para a camada atual, considerando os pesos das camada seguinte
        propagatedErrorSignals = np.dot(
            nextLayerErrorSignals, nextLayerWeights[:, 1:])
        
        # Calcula o errorSignal da camada atual, considerando a derivada da função de ativação
        errorSignals = propagatedErrorSignals * activationD

    # calcula o gradiente da camada atual e multiplica pela learningRate
    gradients = np.outer(errorSignals, layerInput) * learningRate

    # obtém os pesos da camada atual após a retropropagação
    newWeights = currentWeights - gradients

    return newWeights, errorSignals
