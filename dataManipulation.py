from sklearn.datasets import load_iris


def retrieve_data():
    X, y = load_iris(return_X_y=True)

    data = {}

    for i in range(len(X)):

        if y[i] == 0:
            flowercl = 'Iris-Setosa'
        elif y[i] == 1:
            flowercl = 'Iris-Versicolour'
        elif y[i] == 2:
            flowercl = 'Iris-Virginica'

        data[i + 1] = {'sepalLen': X[i][0],
                       'sepalWid': X[i][1],
                       'petalLen': X[i][2],
                       'petalWid': X[i][3],
                       'flowercl': flowercl}

    return data
