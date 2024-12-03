# ine5664

## 1. Introdução

Repositório para o Projeto Final da disciplina INE5664 (Aprendizado de Máquina) da UFSC, desenvolvido pelos alunos:

_Matheus Antunes Monteiro_
_Bruno Duarte de Borja_
_Leandro do Nascimento_

O projeto consiste na implementação de um modelo de rede neural e seu treinamento para 3 tarefas: regressão, classificação binária e classificação multiclasses.

A implementação contempla:

- A estrutura da rede (pesos, camadas, neurônios, etc)
- 3 funções de ativação (e respectivas derivadas): Logística, Relu e Softmax (arquivo [activation.py](https://github.com/MonteiroMath/ine5664/blob/main/activation.py))
- 3 funções de perda (e respectivas derivadas): Erro Quadrado Médio, Entropia Cruzada Binária e Entropia Cruzada Categórica (arquivo [cost.py](https://github.com/MonteiroMath/ine5664/blob/main/cost.py))
- Algoritmo de retropropagação (arquivo [backpropagation.py](https://github.com/MonteiroMath/ine5664/blob/main/backpropagation.py))
- Otimização por gradiente descendente (utilizado na retropropagação)

Foram utilizadas as bibliotecas NumPY para operações matemática e scikit-learn para obtenção de métricas e separação de amostras para treinamento e validação. Não foram utilizadas bibliotecas de alto nível que implemente, completa ou parcialmente, qualquer requisito do trabalho.

Abaixo, serão apresentados brevemente os principais arquivos do projeto e orientações para uso.

## 2. Principais arquivos do projeto

Os principais arquivos desenvolvidos no projeto são:

1. [activation.py](https://github.com/MonteiroMath/ine5664/blob/main/activation.py) : implementa as funções de ativação e respectivas derivadas
2. [cost.py](https://github.com/MonteiroMath/ine5664/blob/main/cost.py): implementa as funções de custo e respectivas derivadas
3. [initLayers.py](https://github.com/MonteiroMath/ine5664/blob/main/initLayers.py): contém a função de inicialização da estrutura da rede (pesos, camadas, neurônios, etc) para treinamento.
4. [train.py](https://github.com/MonteiroMath/ine5664/blob/main/train.py): contém a função de treinamento da rede neural
5. [rna.py](https://github.com/MonteiroMath/ine5664/blob/main/rna.py): contém a rede neural em si, ou seja, contém funções para realizar a etapa de ForwardPass no processo de treinamento e para realizar predições após o treinamento.
6. [backpropagation.py](https://github.com/MonteiroMath/ine5664/blob/main/backpropagation.py): contém o algoritmo de retropropagação. Há duas funções: backpropagation, que controla o processo para toda a rede, e backpropagateLayer, que realiza as operações de retropropagação em uma camada de cada vez.

Abaixo, os arquivos e funções serão detalhados.

### 2.1 [activation.py](https://github.com/MonteiroMath/ine5664/blob/main/activation.py)

Implementa as funções de ativação e respectivas derivadas, quais sejam:

- Logística: através das funções sigmoid(x) e sigmoidDerivative(x)
- Relu: através das funções ReLU(x) e ReluDerivative(x)
- Softmax: através das funções softmax(x) e softmaxDerivative(x)
- Identidade: através das funções identity e identityDerivative

Também é criado um dicionário relacionando o nome das funções a uma tupla que contém a função e sua derivada. O usuário final deve informar ao software apenas o nome da função que será utilizada em cada camada quando da inicialização das camadas (initLayers). As importações necessárias serão realizadas pelo software.

### 2.2 [cost.py](https://github.com/MonteiroMath/ine5664/blob/main/cost.py)

Implementa as funções de custo e respectivas derivadas, quais sejam:

- Erro Quadrado Médio: Através das funções meanSquaredError e mseDerivative
- Entropia Cruzada Binária: Através das funções binaryCrossEntropy e binaryEntropyDerivative
- Entropia Cruzada Categórica: Através das funções categoricalCrossEntropy e categorialEntropyDerivative

Também é criado um dicionário relacionando o nome das funções a uma tupla que contém a função e sua derivada. O usuário final deve informar ao software apenas o nome da função que será utilizada pela rede. As importações necessárias serão realizadas pelo software.

### 2.3 [initLayers.py](https://github.com/MonteiroMath/ine5664/blob/main/initLayers.py)

Contém a função de inicialização da estrutura da rede (pesos, camadas, neurônios, etc) para treinamento. A função possui a seguinte assinatura:

def initLayers(layers, attrNum):

Os argumentos são:

- _layers_: um array de tuplas. Cada tupla representa uma camada da rede neural e possui o formato (númeroDeNeurônios, idFunçãoDeAtivação). Por exemplo, para uma rede neural com duas camadas ocultas com, respectivamente, 3 e 5 neurônios, e uma camada de output de 1 neurônio que utiliza a função Relu nas camadas ocultas e Logística na camada de output, o argumento layer deve possuir o formato:

[
(3, "RELU"), # Camada oculta 1
(5, "RELU"), # Camada oculta 2
(1, "SIGMOID") # Camada de output
]

São considerados valores válidos para funções de ativação:

SIGMOID
RELU
SOFTMAX
IDENTITY

- _attrNum_ : o número de atributos do input. É utilizado para definir quantos pesos cada neurônio da primeira camada deve possuir (attrNum + 1, considerando o viés).

Por exemplo, para um input com observações como (0.7, 0.5, 0.9), temos 3 atributos a serem considerados pela rede, portanto attrNum deve receber o valor 3.

O valor de retorno é:

- initialized_layers: uma lista de dicionários. Cada dicionário representa os parâmetros de uma camada. Os dicionários possuem as propriedades "weights", que representam os pesos da camada, "activation", representando a função de ativação da camada, e "derivation", representando a derivada dessa função.

### 2.4 [train.py](https://github.com/MonteiroMath/ine5664/blob/main/train.py)

Implementa a função para treinamento da rede neural. A função tem a seguinte assinatura:

def train(epochs, learningRate, layers, observations, costF):

Os argumentos são:

- _epochs_: O número de epochs utilizadas para treinamento
- _learningRate_: a taxa de aprendizado da rede
- _layers_: Uma lista de dicionários que informa a estrutura da rede (número de camadas, número de neurônios, funções de ativação de cada camada). Espera-se que o usuário utilize a lista initialized_layers, gerada pela função initLayers.
- _observations_: Uma lista contendo as observações para treinamento.
- _labels_: Uma lista contendo as labels/target values das observações
- _costF_: identificador para a função de custo. São considerados identificadores válidos para a função de custo:

MSE
BINARY_ENTROPY
CATEGORICAL_ENTROPY

Valor de retorno:

- _layers_: lista contendo a estrutura da rede após o treinamento. É a mesma lista recebida como input, porém com os pesos ajustados.

Foram incluídos comentários no arquivo explicando o funcionamento interno da função, razão pela qual ela não será explicada aqui.

### 2.5 [rna.py](https://github.com/MonteiroMath/ine5664/blob/main/rna.py)

Contém a rede neural em si, ou seja, contém funções para realizar a etapa de ForwardPass no processo de treinamento e para realizar predições após o treinamento. O arquivo inclui as funções prepareInput, forwardPass e rna, explicadas abaixo

#### 2.5.1 prepareInput:

Função simples que inclui o valor 1 no início das listas de atributos da observação para acomodar o viés.

#### 2.5.2 forwardPass

Realiza a operação de forward pass para uma camada da rede neural. Essa função é responsável por realizar o produto vertorial dos pesos e inputs, combinando-os linearmente, e por utilizar o resultado dessa operação na função de ativação. Em essencia, executa as operações dos neurônios.

Possui a assinatura:

def forwardPass(input, weights, activationFunction):

Os argumentos são:

- _input_: Array contendo as entradas da camada (atributos do input ou ativações da camada anterior)
- _weights_: weights: matriz contendo os pesos de cada neurônio da camada
- _activationFunction_: a função de ativação utilizada na camada

Valores de retorno:

- _activation_: os valores calculados para a ativação dos neurônios

- _combination_: os valores calculados na combinação linear dos neurônios

Os valores de combination são retornado para serem posteriormente utilizados pelo algoritmo de retropropagação

#### 2.5.3 rna

Representa a rede neural, ou seja, implementa a etapa de forward propagation da rede neural durante o treinamento e, quando utilizado com pesos ajustados, realiza predições.

Essa função itera por todas as camadas, calculando suas ativações e repassando para a camada seguinte. Para maiores detalhas da implementação interna, ver
os comentários no arquivo rna.py

Argumentos:

- _input_: Array contendo os atributos da observação para processamento

- _layers_: Uma lista de dicionários que informa a estrutura da rede (número de camadas, número de neurônios, funções de ativação de cada camada). Na fase de treinamento, espera-se que o usuário utilize a lista initialized_layers, gerada pela função initLayers. Após o treinamento, o usuário deve utilizar a lista "layers" gerada pela função train.

Valor de retorno:

- _activations_: os valores de ativação calculados pela camada de output

Além dos valores de retorno, a função inclui em cada camada representada no input _layers_ a propriedade "intermediate", que contém os valores intermediários produzidos na camada durante a forward pass. Esses valores são: o input de cada camada e as combinações lineares produzidas nela, da seguinte maneira:

layerParams["intermediate"] = (layerInput, combinations)

Isso torna essas valores acessíveis pela função train, que as repassa para o algoritmo de retropropagação.

### 2.6 [backpropagation.py](https://github.com/MonteiroMath/ine5664/blob/main/backpropagation.py)

Contém o algoritmo de retropropagação. Há duas funções: backpropagation, que controla o processo para toda a rede, e backpropagateLayer, que realiza as operações de retropropagação em uma camada de cada vez.

#### 2.6.1 backpropagation

Implementa a operação da retropropagação para toda a rede. Essa função itera por todas as camadas em sentido inverso (da camada de output para trás), propagando o erro da camada de output através delas. Para isso, faz uso da função backpropagateLayer, que realiza as operações específicas para cada camada.

Para maiores detalhes de implementação interna, ver os comentários do arquivo [backpropagation.py](https://github.com/MonteiroMath/ine5664/blob/main/backpropagation.py)

Argumentos:

- _layers_: lista contendo pesos e funções de ativação/derivadas, bem como os valores intermediários produzidos na forwardPass. É a lista produzida pela função initLayers e atualizada pela função rna.
- _costD_: derivada da função de custo, já calculada pela função train.
- _learningRate_: parâmetro de learningRate definido inicialmente

Valor de retorno:

- _adjustedWeights_: os pesos atualizados após a retropropagação. A função train os recebe e atualiza os pesos da lista layers.

#### 2.6.2 backpropagateLayer

Realiza a operação de backpropagação em uma camada. Essa é a função que efetivamente:

- Calcula a derivada da função de ativação de cada camada
- Calcula o gradiente descendente
- Utiliza o gradiente descendente e a taxa de aprendizado para calcular o valor de ajuste a realizar nos pesos
- Calcula os novos pesos para os neurônios da camada.

Todas essas operações são realizadas de forma vetorial. Para maiores detalhes de implementação interna, ver os comentários do arquivo [backpropagation.py](https://github.com/MonteiroMath/ine5664/blob/main/backpropagation.py)

Argumentos:

- _weights_: tupla contendo os pesos da layer atual e os pesos da layer seguinte
- _intermediateValues_: tupla contendo os valores intermediários relevantes para a camada (inputs recebidos e combinações produzidas)
- _learningRate_: parâmetro de learningRate
- _costD_: valor calculado para a derivada da função de custo
- _activationDerivative_: função para cálculo da derivada da função de ativação
- _nextLayerErrorSignals_: Error signals da camada seguinte

Valores de retorno:

- _newWeights_: os pesos ajustados após o processo de retropropagação
- _errorSignals_: os sinais de erro produzidos na camada, que serão propagados para a camada anterior na próxima etapa

## 3. Instruções de uso básicas

Abaixo, seguem instruções básicas para uso do código produzido neste projeto.

Observe que são instruções básicas visando explicar como utilizar a rede neural produzida, mas não inclui cuidados importantes em projetos de aprendizado de máquina, como reserva de valores não utilizados no treinamento para validação da rede (até porque utiliza-se, no exemplo, um dataset minúsculo apenas como exemplo)

### 3.1 Para treinamento

- Inicializar uma lista com o número de camadas, o número de neurônios de cada camada e a função de ativação de cada camada, conforme explicado em initLayers.py

Exemplo:

```python
layers = [
    (2, 'RELU'),
    (1, 'SIGMOID')
]
```

- Definir a função de custo, o número de épocas para treinamento e a taxa de aprendizado

Exemplo:

```python
costF = "BINARY_ENTROPY"
EPOCHS = 1000
LEARNING_RATE = 0.1
```

- Chamar a função intLayers para inicializar a estrutura da rede com pesos aleatórios

Exemplo:

```python
layers = initLayers(layers, 2)
```

- Preparar as observações e labels

Exemplo:

```python
observations = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

labels = np.array([0, 1, 1, 1])
labels = labels.reshape(-1, 1)

```

- Chamar a função train com todos esses parâmetros e obter os valores ajustados da rede

Exemplo:

```python
trainedParams = train(EPOCHS, LEARNING_RATE, layers, observations, costF)
```

Exemplo completo da fase de treinamento:

```python
from train import train
from initLayers import initLayers

from rna import rna

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

trainedParams = train(EPOCHS, LEARNING_RATE, layers, observations, labels, costF)

```

### 3.2 Para obter predições

- Basta chamar a função RNA repassando os parâmetros ajustados e o input para predição. Considerando os valores definidos no exemplo da fase de treinamento, segue exemplo:

```python

for observation, label in zip(observations, labels):

  activations = rna(observation, trainedParams)

  prediction = 1 if activations > 0.5 else 0

  print(f"For input {observation} with {label} the prediction was {prediction} \n")


```

## 4 Jupyter Notebook

### 4.1 Regressão

### 4.2 Classificação Binária

### 4.3 Classificação multiclasses
