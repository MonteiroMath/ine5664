import numpy as np
from initLayers import initLayers
from train import train
from rna import rna
from utils import normalize, splitData, accuracy

# Carrega dados do dataset:
DATASET_PATH = './data/heart.csv'
data = np.genfromtxt(DATASET_PATH, delimiter=',', skip_header=1)

# Separa features e labels:
features = data[:, :-1]
labels = data[:, -1:]

# Separa dados para treinamento e para validação:
featuresTrain, featuresVal, labelsTrain, labelsVal = splitData(features, labels)

# Normaliza features:
normalized_features, feature_scaler = normalize(featuresTrain)

# Define a estrutura da rede, funções de ativação, funções de custo, épocas e learning rate:
# Obtém a quantidade de neurons da camada de input
INPUT_NEURONS = len(normalized_features[0])

# Define a estrutura da rede e funções de ativação
layers = [
    (INPUT_NEURONS, 'RELU'),
    #(INPUT_NEURONS, 'RELU'),
    (1, 'SIGMOID')
]

# Definição função de custo, épocas e learning rate
COSTF = "BINARY_ENTROPY"
EPOCHS = 50
LEARNING_RATE = 0.001

# Inicializa os pesos e as funções das camadas:
layers = initLayers(layers, INPUT_NEURONS)

# Dispara o treinamento da rede:
trainedParams = train(EPOCHS, LEARNING_RATE, layers, normalized_features, labelsTrain, COSTF)

# Normaliza as features para validação:
test_normalized_features = feature_scaler.transform(featuresVal)

# Testa a rede treinada:
predictions = []
for i, observation in enumerate(test_normalized_features):

  input = observation
  prediction = rna(input, trainedParams)

  #one_hot_prediction = np.zeros_like(prediction)
  #one_hot_prediction[np.argmax(prediction)] = 1

  #predictions.append(one_hot_prediction)
  predictions.append(prediction)

# Obtém métricas de avaliação:
#metrics = evaluate_multiclass_with_one_hot(labelsVal, predictions)

accuracy = accuracy(np.round(np.array(predictions)), np.round(np.array(labelsVal)))
print(" accuracy: ", accuracy)


# plt.plot()
# plt.title("classificacao binaria - teste \n custo usando entropia cruzada binaria")
# plt.xlabel("epoca")
# plt.ylabel("custo")
# plt.show()
#
