#!/usr/bin/env python
# inter_transf.py
# Author/Professor: Fabrício Olivetti de França
# Author/Student: Maira Zabuscha de Lima

import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV

# ignoramos os resultados NaN das funções pois vamnos zera-los
np.seterr(invalid='ignore')

def importaDados(fname):
    # carrega os dados do arquivo fname e retorna X, y
    dataset = np.loadtxt(fname, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return (X, y)

def transformData(X, rede):
    # recebe uma matriz X e uma rede de expoentes
    # retorna os dados transformados
    # substitui todos os valores NaN e Inf para 0
    n_rows = X.shape[0]
    n_inter = rede.shape[1]
    layers = np.ndarray((n_rows, 3*n_inter))
    for i in range(0, n_inter*3, 3):
        power = X**rede[:,int(i/3)]
        layers[:,i] = np.prod(power,axis=1) # id
        layers[:,i+1] = np.cos(layers[:,i]) # cos
        layers[:,i+2] = np.sqrt(layers[:,i]) # sqrt
    cols = np.any(np.isnan(layers), axis=0)
    layers[:, cols] = 0
    return layers

def fit(X, y, n_inter):
    # Cria a camada de expoentes da rede com n_inter neurônios
    # Aplica a função transformData em X_train utilizando essa rede
    # Divida a base entre treino e validação
    # Aplique o LassoCV e LassoLarsCV, verifique na validação o que retorna o menor erro
    # Retorne a rede e o modelo de menor erro
    # expoentes aleatorios de 0 a 2
    n_inputs = X.shape[1]
    exponents = np.random.randint(0,3,size=(n_inputs, n_inter))
    X_transf = transformData(X, exponents)
    X_train, X_test, y_train, y_test = train_test_split(X_transf, y, test_size=0.3, random_state=1)
    models = {'lassoCV':LassoCV(max_iter=5e4, cv=3), 'lassoLarsCV':LassoLarsCV(max_iter=5e4, cv=3)}
    rmse = {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse[key] = np.sqrt(mean_squared_error(y_test, y_pred))
        print(key, rmse[key]) # so pra conferir se eu to fazdo certo
    print('min', min(rmse, key=rmse.get)) # so pra conferir se eu to fazdo certo
    return exponents, models[min(rmse, key=rmse.get)]

def predict(X_test, rede, modelo):
    # Aplica transformData em X_test usando a rede
    # Aplique o método predict de modelo na base transformada e armazena a saída em y_hat
    # Retorna y_hat
    X_transf = transformData(X_test, rede)
    y_hat = modelo.predict(X_transf)
    return y_hat

def main(file='airfoil'):
    NINTER = 5
    print('n inter', NINTER)

    fileTrain = 'datasets/' + file + '-train-0.dat'
    X_train, y_train = importaDados(fileTrain)
    rede, modelo = fit(X_train, y_train, NINTER)

    fileTest = 'datasets/' + file + '-test-0.dat'
    X_test, y_test = importaDados(fileTest)
    y_pred = predict(X_test, rede, modelo)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('rmse', rmse)

if __name__ == "__main__":
    file = 'airfoil'
    if len(sys.argv) > 1:
        file = sys.argv[1]
    main(file)
