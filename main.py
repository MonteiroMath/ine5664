from train import train
from initLayers import initLayers
import numpy as np

# Lista no formato [(neurons, activation), (neurons, activation), (neurons, activation)]
# Cada tupla representa uma camada
layers = [
    (2, 'RELU'),
    # (2, 'SIGMOID'),
    (1, 'SIGMOID')
]

costF = "BINARY_ENTROPY"

layers = initLayers(layers, 2)

EPOCHS = 1000
LEARNING_RATE = 0.1

observations = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

labels = np.array([0, 1, 1, 1])
labels = labels.reshape(-1, 1)


train(EPOCHS, LEARNING_RATE, layers, observations, labels, costF)
