from train import train
from initLayers import initLayers

# Lista no formato [(neurons, activation), (neurons, activation), (neurons, activation)]
# Cada tupla representa uma camada
layers = [
    (2, 'SIGMOID'),
    # (2, 'SIGMOID'),
    (1, 'SIGMOID')
]

costF = "MSE"

layers = initLayers(layers, 2)

EPOCHS = 1000
LEARNING_RATE = 0.1

observations = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


train(EPOCHS, LEARNING_RATE, layers, observations, costF)
