import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

path = input("Enter path:")
col_names = input("Enter dependent/independent variables:")

def extract(path, col_names):
    df = pd.read_csv(path)
    col_names = col_names.split(' ')
    df = df[col_names]
    current_row = None
    old_row = None
    sum_row = []
    n = 0
    Y = []
    X = []
    for index, row in df.iterrows():
        if index == 0:
            old_row = row
        else:
            if row['dayofweek'] == old_row['dayofweek']:
                n += 1
                old_row = row
            else:
                x = []
                x.append(df[:n])
        X.extend(x)
            # n = 1
            # sum_row = row[col_names]
        current_row = row


extract(path, col_names)
#/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv
# SMA_50 SMA_20 dayofweek


    #
    # for col_name in range(0, len(col_names)):