#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random;

LR = 0.01
N_EPOCHS = 10000

class SlyLogRegression():

    def __init__(self, lr=LR, weight = [], n_epochs=N_EPOCHS):
        self.weight = weight
        self.n_epochs = N_EPOCHS
        self.lr = lr

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _get_selection(self, X, y, is_sgd):
        x_train = X
        y_train = y
        m = len(x_train)
        if(is_sgd):
            r = random.randint(0,len(y) - 1)
            x_train = X[r]
            y_train = y_train[r]
            m = 1
        return x_train, y_train, m

    def fit(self, X, y, is_sgd=False):
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)
        theta = []
        for class_marker in np.unique(y):
            y_copy = np.where(y == class_marker, 1, 0)
            w = np.ones(X.shape[1])
            theta.append(w)
            for i in range(len(y)):
                x_train, y_train, m = self._get_selection(X, y_copy, is_sgd)

                hypothesis = self._sigmoid(x_train.dot(theta[class_marker]))
                loss = hypothesis - y_train
                gradient = np.dot(x_train.transpose(), loss) / m

                theta[class_marker] = theta[class_marker] - self.lr * gradient
        self.weight = theta

    def predict(self, X):
        result =  []
        X = np.array(X)
        X = np.insert(X, 0, 1, axis=1)

        if len(self.weight) == 0:
            print('Weight is empty')
            return

        for i in range(len(X)):
            pre = []
            for j in range(len(self.weight)):
                pre.append(self._sigmoid(X[i].dot(self.weight[j])))
            result.append(pre.index(max(pre)))
        return np.array(result)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
