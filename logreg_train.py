#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from libs import snackbar, SlyLogRegression
from read import read_csv


def preprocess():
    if len(sys.argv) == 1:
        snackbar("Set file name", 'error')
    df = read_csv(sys.argv[1])
    df = df.dropna()
    targert_col = 'Hogwarts House'
    if (targert_col not in df.keys()):
        snackbar("Your dataset don't have Hogwarts House", 'error')
    if(df.shape[0] == 0 or df.shape[1] == 0):
        snackbar("Your dataset is empty", 'error')
    y = df[targert_col]
    labelTransform = LabelEncoder()
    labelTransform.fit(y)
    y = labelTransform.transform(y)

    X = df[df.describe().columns[1:]]
    X = X.fillna(X.mean())
    X = (X - X.mean()) / X.std()

    model = SlyLogRegression()
    model.fit(X, y)
    weight_df = pd.DataFrame(np.array(model.weight).T, columns=labelTransform.classes_)
    weight_df.to_csv('weight.csv', index_label="Index")


def main():
    preprocess()


if __name__ == '__main__':
    main()
