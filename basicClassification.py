from train import train
from initLayers import initLayers

from rna import rna

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
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


trainedParams = train(EPOCHS, LEARNING_RATE, layers, observations, costF)



for observation in observations:

  input, label = observation

  activations = rna(input, trainedParams)

  prediction = 1 if activations > 0.5 else 0

  print(f"For input {input} with {label} the prediction was {prediction} \n")



