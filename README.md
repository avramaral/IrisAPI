# <img src="./static/flower.svg" width = "25" height = "25" style = "opacity: 0.75"/> Iris API

## Modelos de classificação para o conjunto de dados [Iris](https://archive.ics.uci.edu/ml/datasets/iris)

Essa API, construída com o framework [Fast API](https://fastapi.tiangolo.com/), fornece duas rotas para lidar com o conjunto de dados desejado:

1. `/data/` &rarr; Aqui, basta fazer uma requisição utilizando o método `GET`. O resultado é serializado em formato `JSON`.
2. `/predict/` &rarr; Aqui, basta fazer uma requisição utilizando o método `POST`. Nesse caso:
    - É possível incluir um parâmetro de query chamado de modelName, associando a ele uma das seguintes opções: `LogisticRegression`, `KNN`, `MultiLayerPerceptron` (a.k.a. "NeuralNetwork") ou `SupportVectorClassifier`. Assim, a rota ficaria `/predict?modelName=<NomeDoModelo>`, com `LogisticRegression` como valor padrão.
    - É necessário incluir no corpo da requisição os valores para as seguintes quantidades: `{ "sepalLen": 0.0, "sepalWid": 0.0, "petalLen": 0.0, "petalWid": 0.0 }`.

### Modelos

Os modelos foram ajustados utilizando a biblioteca Scikit-learn. O código para o ajuste de cada modelo é mostrado abaixo:

-   `LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class = 'ovr', max_iter = 1000).fit(X, y)
```

-   `KNN`

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 5).fit(X, y)
```

-   `MultiLayerPerceptron`

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes = (32, 32, 32), max_iter = 1000).fit(X, y)
```

-   `SupportVectorClassifier`

```python
from sklearn.svm import SVC

model = SVC(decision_function_shape = 'ovr', max_iter = 1000).fit(X, y)
```

### Exemplos

1. No caso de uma requisição feita a partir da rota `/data/`, utilizando o método `GET`, vamos obter:

```json
{
    "1": {
    "sepalLen": 5.1,
    "sepalWid": 3.5,
    "petalLen": 1.4,
    "petalWid": 0.2,
    "flowercl": "Iris-Setosa"
    },
    "2": {
    "sepalLen": 4.9,
    "sepalWid": 3,
    "petalLen": 1.4,
    "petalWid": 0.2,
    "flowercl": "Iris-Setosa"
    },
        [...]
```

2. No caso de uma requisição feita a partir da rota `/predict?modelName=LogisticRegression`, utilizando o método `POST`, e enviando no corpo da requisição as quantidades `{ "sepalLen": 7.2, "sepalWid": 2.7, "petalLen": 6, "petalWid": 1.8 }`, vamos obter:

```json
{
    "sepalLen": 7.2,
    "sepalWid": 2.7,
    "petalLen": 6,
    "petalWid": 1.8,
    "modelName": "LogisticRegression",
    "prediction": 2
}
```

Aqui, o que nos interessa é o valor `{ "prediction": 2 }`, que determina que a flor que possui as dimensões informadas é da espécie "Iris-Virginica". Para referência, se `prediction` for `0`, então a planta é da espécie "Iris-Setosa"; se for `1` é "Iris-Versicolour", e, como vimos, se for `2`, é "Iris-Virginica".
