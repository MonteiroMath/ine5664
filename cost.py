import numpy as np


def meanSquaredError(predictions, labels):
    # Usar para regressão
    return np.mean((predictions - labels) ** 2)


def mseDerivative(prediction, label):
    # Derivada da mse para uma observação
    return 2 * (prediction - label)


def binaryCrossEntropy(predictions, labels):
    # Usar para classificação binária

    epsilon=1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    return -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))


def binaryEntropyDerivative(prediction, label):

    epsilon=1e-15
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return (prediction - label) / (prediction * (1 - prediction))


def categoricalCrossEntropy(predictions, labels):
    # usar para multiclassificação
    return -np.sum(labels * np.log(predictions)) / predictions.shape[0]


def categoricalEntropyDerivative(predictions, labels):
    pass


costFunctions = {
    "MSE": (meanSquaredError, mseDerivative),
    "BINARY_ENTROPY": (binaryCrossEntropy, binaryEntropyDerivative),
    "CATEGORICAL_ENTROPY": (categoricalCrossEntropy, categoricalEntropyDerivative)
}
