import numpy as np


def meanSquaredError(predictions, labels):
    # Usar para regressão
    return np.mean((predictions - labels) ** 2)

def binaryCrossEntropy(predictions, labels):
    # Usar para classificação binária

    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

def categoricalCrossEntropy(predictions, labels):
    # usar para multiclassificação
    return -np.sum(labels * np.log(predictions)) / predictions.shape[0]