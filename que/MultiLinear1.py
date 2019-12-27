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
    group_row = pd.DataFrame(columns=[col_names])
    new_df =  pd.DataFrame(columns=[col_names])
    for index, row in df.iterrows():
        if index == 0:
            old_row = row
            group_row.append(row)
            n = 1
        else:
            if row['dayofweek'] == old_row['dayofweek']:
                old_row = row
                group_row.append(row)
                n += 1
            else:
                old_row = row
                new_df.append(sum(group_row))
                n = 1
                del temp
    Y = new_df.iloc[:, :1]
    X = new_df.iloc[:, 1:]
    return X, Y

extract(path, col_names)
#/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv
# SMA_50 SMA_20 dayofweek


    #
    # for col_name in range(0, len(col_names)):