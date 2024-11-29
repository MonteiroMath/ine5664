import numpy as np


def sigmoid(x):
    #Função de ativação usada para normalizar valores em um intervalo de 0 a 1
    return 1/(1 + np.exp(-x))

def sigmoidDerivative(x):
    #Calcula a derivada da função sigmoide, útil para o backpropagation
    sigmoidResult = sigmoid(x)
    return sigmoidResult * (1 - sigmoidResult)


def identity(x):
    #Função de ativação que não modifica a saída do neurônio, útil em tarefas de regressão 
    return x

def identityDerivative(x):
    #Derivada da função identidade
    return np.ones_like(x)

def ReLU(x):
    #Função de ativação usada para introduzir não-linearidade. Muito utilizada para neurônios de camadas ocultas.
    return np.maximum(0, x)

def ReluDerivative(x):
    #Calcula a derivada de ReLU
    return np.where(x > 0, 1, 0)

def softmax(x):
    # usar para classificação multi-classes

    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmaxDerivative(x):
    # Em razão da fórmula de cálculo para cost da categoricalEntropy, não é necessário calcular a derivada da softmax
    return 1


activationFunctions = {
    "SIGMOID": (sigmoid, sigmoidDerivative),
    "RELU": (ReLU, ReluDerivative),
    "SOFTMAX": (softmax, softmaxDerivative),
    "IDENTITY": (identity, identityDerivative)
}