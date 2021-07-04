#!/usr/bin/env python3
import sys
import pandas as pd
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

    # Remove nan row
    df.dropna(inplace=True)
    Y = df[targert_col]

    # Normalize
    df_number = df.select_dtypes(include=['float64', 'int64'])
    df_normal = (df_number - df_number.mean()) / df_number.std()

    name_courses = list(df_normal.keys())
    name_courses.remove('Index')
    size_plot = round((len(name_courses)) ** 0.5)
    df_normal[targert_col] = Y
    if df_normal.shape[0] == 0 or df_normal.shape[1] == 0:
        snackbar("Empty dataset", 'error')
    _, axes = plt.subplots(nrows=size_plot, ncols=size_plot, figsize=(20, 20), sharey=True)
    plots = axes.flatten()
    i = 0
    for name in name_courses:
        if(name != targert_col):
            sns.histplot(ax=plots[i], data=df_normal, x=name, hue=targert_col, kde=True, common_norm=True, stat="density")
            plots[i].set_xlabel(name, fontsize=14)
            plots[i].grid()
            i += 1
    plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.35)
    plt.show()


    norm_course = 'Care of Magical Creatures'
    _ = plt.figure(figsize=(10, 10))
    ax = sns.histplot(data=df_normal, x=norm_course, hue=targert_col, kde=True, common_norm=True, stat="density")
    ax.set_xlabel(norm_course, fontsize=14)
    plt.show()

def main():
    preprocess()


if __name__ == '__main__':
    main()
