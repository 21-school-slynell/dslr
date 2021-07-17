#!/usr/bin/env python3
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from libs import snackbar
from read import read_csv


def preprocess():
    if len(sys.argv) == 1:
        snackbar("Set file name", 'error')
    df = read_csv(sys.argv[1])
    targert_col = 'Hogwarts House'
    if (targert_col not in df.keys()):
        snackbar("Your dataset don't have Hogwarts House", 'error')

    plt.figure(figsize=(10, 10))
    target_col = 'Hogwarts House'
    sns.pairplot(df, hue=target_col)
    plt.show()


def main():
    preprocess()


if __name__ == '__main__':
    main()
