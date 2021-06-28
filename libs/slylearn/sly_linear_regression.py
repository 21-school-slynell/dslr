#!/usr/bin/env python3
import numpy as np

SPEED = 0.1
EPS = 0.000000001
N_EPOCHS = 10000

class SlyLinearRegression():

    def __init__(self, b = 0, k = 0):
        self.b = b;
        self.k = k;

    def normalize(self, arr):
        return (arr - min(arr)) / (max(arr) - min(arr))

    def MSE(self, y, y_predict):
        return ((y - y_predict) ** 2).mean()

    def predict(self, x):
        return self.b + x * self.k

    def denorm(self, X, Y):
            x_min = X.min()
            y_min = Y.min()
            x_max = X.max()
            y_max = Y.max()

            k = (y_max - y_min) * self.k / (x_max - x_min)
            b = y_min + ((y_max - y_min) * self.b) + k * (1 - x_min)
            return b, k

    def R2(self, y, y_predic):
        y_mean = y.mean()
        return 1 - ((y_predic - y) ** 2).sum() / ((y - y_mean) ** 2).sum()

    def fit(self, x, y, n_epoche = N_EPOCHS, speed = SPEED ):
        x_normal = self.normalize(x)
        y_normal = self.normalize(y)

        y_predict = self.predict(x_normal)
        error = self.MSE(y_normal, y_predict);
        stop = error

        i = 0
        while(stop > EPS and i < n_epoche):

            error_last = error

            b_grad = (y_predict - y_normal)
            k_grad = (b_grad * x_normal)
            self.b = self.b - speed * b_grad.mean()
            self.k = self.k - speed * k_grad.mean()

            y_predict = self.predict(x_normal)
            error = self.MSE(y_normal, y_predict);
            stop = abs(error_last - error)
            i += 1;
        self.b, self.k = self.denorm(x, y)

    def score(self, x, y):
        x = np.array(x)
        y = np.array(y)
        y_predict = self.predict(x)
        error = self.R2(y, y_predict)
        return error
