import numpy as np


def meanSquaredError(predictions, labels):
    # Função de custo erro quadrático. Usar para regressão
    return np.mean((predictions - labels) ** 2)


def mseDerivative(prediction, label):
    # Derivada da mse para uma observação
    return 2 * (prediction - label)


def binaryCrossEntropy(predictions, labels):
    # Calcula o erro de entropia cruzada. Usar para classificação binária

    epsilon=1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon) #limita valores de prediction para evitar logaritmos de 0 ou 1, que resultariam em erros.

    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))


def binaryEntropyDerivative(prediction, label):
    #Calcula a derivada da entropia cruzada binária
    epsilon=1e-15
    prediction = np.clip(prediction, epsilon, 1 - epsilon) #limita valores de prediction para evitar logaritmos de 0 ou 1, que resultariam em erros.
    return (prediction - label) / (prediction * (1 - prediction))


def categoricalCrossEntropy(predictions, labels):
    # usar para multiclassificação
    return -np.sum(labels * np.log(predictions)) / predictions.shape[0] # dividir por predictions.shape[0] tem o intuito de normalizar o valor da perda pelo número de amostras, tornando a perda independente do tamanho do batch


def categorialEntropyDerivative(predictions, labels):
    pass


costFunctions = {
    "MSE": (meanSquaredError, mseDerivative),
    "BINARY_ENTROPY": (binaryCrossEntropy, binaryEntropyDerivative),
    "CATEGORICAL_ENTROPY": (categoricalCrossEntropy, categorialEntropyDerivative)
}
