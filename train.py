import numpy as np
from rna import rna
from cost import costFunctions
from backpropagation import backpropagation


def train(epochs, learningRate, layers, observations, costF):
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

    # extrai a função de custo e sua derivada
    costFunction, costDerivative = costFunctions[costF]

    # itera pela quantidade de épocas definida
    for n in range(epochs):

        predictions = []

        # itera pelas observações
        for observation in observations:

            # extrai atributos e label das observações
            attributes, label = observation
            attributes = np.array(attributes)

            # Repassa os atributos e parâmetros das camadas para a rede neural e obtém uma predição e os valores intermediários produzidos
            prediction, intermediateValues = rna(attributes, layers)
            predictions.append(prediction)

            # invoca a função de custo
            cost = costFunction(prediction, label)
            print(prediction)

            # Calcula a derivada do custo para a observação corrente
            costD = costDerivative(prediction, label)

            # Obtém os pesos ajustados através da retropropagação
            adjustedWeights = backpropagation(
                layers, intermediateValues, costD, learningRate)
            
            # Atualiza os valores dos pesos usando os valores ajustados
            layers[0] = adjustedWeights
