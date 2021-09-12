#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from libs import snackbar, SlyLogRegression
from read import read_csv


def preprocess():
    if len(sys.argv) == 1:
        snackbar("Set file name", 'error')
    df = read_csv(sys.argv[1])
    col = list(df.describe().columns)

    targert_col = 'Hogwarts House'
    col.remove("Index")
    col.remove(targert_col)
    X = df[col]
    X = X.fillna(X.mean())
    X = (X - X.mean()) / X.std()
    if(df.shape[0] == 0 or df.shape[1] == 0 or X.shape[0] == 0 or X.shape[1] == 0):
        snackbar("Your dataset is empty", 'error')
    weight = pd.read_csv('./weight.csv', index_col="Index")
    if(X.shape[1] + 1 != weight.shape[0] or weight.shape[1] == 0):
        snackbar("Your weight is empty or incorrect", 'error')

    labelTransform = LabelEncoder()
    labelTransform.fit(weight.columns)

    model = SlyLogRegression(lr=0.01, weight=np.array(weight.T))
    result = labelTransform.inverse_transform(model.predict(X))
    df_result = pd.DataFrame(result, columns=[targert_col])
    df_result.to_csv('./houses.csv', index_label="Index")

def main():
    preprocess()

if __name__ == '__main__':
    main()
