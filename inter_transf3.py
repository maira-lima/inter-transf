#!/usr/bin/env python
# inter_transf.py
# Author/Professor: Fabrício Olivetti de França
# Author/Student: Maira Zabuscha de Lima
# # Rede Neural de Múltiplas Camadas para Regressão Simbólica

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, LassoLarsCV
import warnings
import sys
import os
import pickle

# ignoramos os resultados NaN das funções pois vamnos zera-los
np.seterr(invalid='ignore')
# nao quero warning de convergência
warnings.filterwarnings('ignore')

def getResults(fname):
    dataset = []
    algoritmo = []
    msre_l = []
    coef_l = []
    rede_l = []
    if os.path.exists(fname):
        fw = open(fname, 'rb')
        dataset, algoritmo, msre_l, coef_l, rede_l = pickle.load(fw)
        fw.close()
    return dataset, algoritmo, msre_l, coef_l, rede_l

def storeResults(dataset, algoritmo, msre_l, coef_l, rede_l, fname):
    fw = open(fname, 'wb')
    pickle.dump((dataset, algoritmo, msre_l, coef_l, rede_l), fw)
    fw.close()

def msre(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def importaDados(fname):
    dataset = np.loadtxt(fname, delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1]
    return (X, y)
    
def geraRede(X, n_inter, inter_min=0, inter_max=3):
    n_inputs = X.shape[1]
    rede = np.random.randint(inter_min, inter_max, size=(n_inputs, n_inter))
    for i in range(0, n_inputs):
        if 0 in X[:,i]:
            rede[i,:] = np.absolute(rede[i,:])
    return rede

def transformData(X, rede):
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
    cols = np.any(np.isinf(layers), axis=0)
    layers[:, cols] = 0
    return layers

def fit(X, y, algoritmo, n_inter, inter_min=0, inter_max=3):
    n_inputs = X.shape[1]
    rede = geraRede(X, n_inter, inter_min, inter_max)
    X_transf = transformData(X, rede)
    if algoritmo == 'lasso':
        modelo = LassoCV(max_iter=5e4, cv=3)
    elif algoritmo == 'lassoLars':
        modelo = LassoLarsCV(max_iter=5e4, cv=3)
    modelo.fit(X_transf, y)
    return rede, modelo

def predict(X, rede, modelo):
    X_transf = transformData(X, rede)
    y_hat = modelo.predict(X_transf)
    return y_hat

def expression(rede, coef, limiar):
    n_inter = rede.shape[1]
    n_inputs = rede.shape[0]
    fun = ['', 'cos', 'sqrt']
    expr = ''
    total = 0
    for i in range(n_inter):
        inter = ''
        for j in range(n_inputs):
            if rede[j,i] != 0:
                inter = inter + f'x{j}**{rede[j,i]}*'
        inter = inter[:-1]
        transf = ''
        for f in range(3):
            k = (i*3)+f
            if np.absolute(coef[k]) > limiar:
                total = total + 1
                transf = transf + f'{coef[k]}*{fun[f]}({inter}) + '
        expr = expr + transf
    expr = expr + f'{coef[k+1]}'
    return expr, total

def evaluate(row, rede, coef, limiar):
    n_inter = rede.shape[1]
    n_inputs = rede.shape[0]
    total = 0.0
    for i in range(n_inter):
        inter = 1.0
        for j in range(n_inputs):
            inter *= np.power(float(row[j]),rede[j,i])
        transf = 0.0
        if np.absolute(coef[i*3+0]) > limiar:
            transf += coef[(i*3)+0]*inter
        if np.absolute(coef[i*3+1]) > limiar:
            transf += coef[(i*3)+1]*np.cos(inter)
        if np.absolute(coef[i*3+2]) > limiar:
            transf += coef[(i*3)+2]*np.sqrt(inter)
        total += transf
    total += coef[-1]
    return total

def main(D, algoritmo, ninter_str, inter_min_str, inter_max_str):
    dataset_l, algoritmo_l, msre_l, coef_l, rede_l = getResults('tests.pkl')
    pastas = ['0', '1', '2', '3', '4']
    ninter = int(ninter_str)
    inter_min = int(inter_min_str)
    inter_max = int(inter_max_str)
    dataset_l.append(D)
    algoritmo_l.append(f'{algoritmo} {ninter} ({inter_min} {inter_max})')
    msre_pastas = []
    coef_pastas = []
    rede_pastas  =[]
    print(f'dataset {D}')
    for pasta in pastas:
        fileTrain = 'datasets/' + D + '-train-' + pasta + '.dat'
        fileTest = 'datasets/' + D + '-test-' + pasta + '.dat'
        X_train, y_train = importaDados(fileTrain)
        X_test, y_test = importaDados(fileTest)
        n = X_train.shape[1]
        rede, modelo = fit(X_train, y_train, algoritmo, ninter*n, inter_min, inter_max+1)
        y_hat = predict(X_test, rede, modelo)
        msre_pastas.append(msre(y_test, y_hat))
        coef = np.append(modelo.coef_, modelo.intercept_)
        coef_pastas.append(coef)
        rede_pastas.append(rede)
    m = len(pastas)
    msre_medio = np.sum(msre_pastas)/m
    msre_l.append(msre_medio)
    min_i = np.argmin(msre_pastas)
    print(f'{D} {algoritmo} {ninter} ({inter_min} {inter_max}) msre: {msre_medio}')
    print('total de coeficientes não-zeros:', np.count_nonzero(coef_pastas[min_i]))
    coef_l.append(coef_pastas[min_i])
    rede_l.append(rede_pastas[min_i])
    storeResults(dataset_l, algoritmo_l, msre_l, coef_l, rede_l, 'tests.pkl')
    print('done')
    
if __name__ == "__main__":
    if len(sys.argv) > 5:
        base = sys.argv[1]
        alg = sys.argv[2]
        ninter_str = sys.argv[3]
        inter_min_str = sys.argv[4]
        inter_max_str = sys.argv[5]
        main(base, alg, ninter_str, inter_min_str, inter_max_str)
    else:
        print('determine a base e ninter')
