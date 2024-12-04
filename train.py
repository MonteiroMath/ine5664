import numpy as np
from rna import rna
from cost import costFunctions
from backpropagation import backpropagation
from matplotlib import pyplot as plt


def train(epochs, learningRate, layers, observations, labels, costF):
    """

    Implementa o algoritmo de treinamento da rede neural.

    Recebe:

        epochs: número de épocas para treinamento

        learningRate: valor do parâmetro de learningRate

        layers: lista com os parâmetros de cada camada da rede. Ver initLayers para mais detalhes.

        observations: lista de tuplas contendo observações para treinamento. Cada tupla tem dois elementos: ([atributosDaObservação, label])
            Exemplo:
            observations = [
                ([0, 0], 0),
                ([0, 1], 1),
                ([1, 0], 1),
                ([1, 1], 1),
                ]

        costF: identificador para função de custo. Valores válidos:
            MSE
            BINARY_ENTROPY
            CATEGORICAL_ENTROPY
    """

    error = []
    
    # extrai a função de custo e sua derivada
    costFunction, costDerivative = costFunctions[costF]

    # itera pela quantidade de épocas definida
    for n in range(epochs):

        predictions = []

        # itera pelas observações
        for observation, label in zip(observations, labels):

            # formata observations e labels como arrays
            observation = np.array(observation)
            label = np.array(label)

            # Repassa os atributos e parâmetros das camadas para a rede neural e obtém uma predição
            prediction = rna(observation, layers)
            predictions.append(prediction)

            # Calcula a derivada do custo para a observação corrente
            costD = costDerivative(prediction, label)

            # Obtém os pesos ajustados através da retropropagação
            adjustedWeights = backpropagation(
                layers, costD, learningRate)

            # Atualiza os valores dos pesos usando os valores ajustados

            for i, layer in enumerate(layers):
                layer["weights"] = adjustedWeights[i]

        # invoca a função de custo
        predictions = np.array(predictions)
        cost = costFunction(predictions, labels)
        print(cost)
        error.append(cost)

    #
    plt.plot(error)
    if (costF == "MSE"):
        plt.title("regressão - treino")
    elif (costF == "BINARY_ENTROPY"):
        plt.title("classificação binária - treino")
    else:
        plt.title("classificação multiclasse - treino")
    plt.xlabel("epoca")
    plt.ylabel("custo")
    plt.show()

    return layers
