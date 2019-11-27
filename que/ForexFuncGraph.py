import pandas as pd
from pandas import DataFrame, Series

def load(paths):
    paths = paths.split(' ')
    df = []
    for path in paths:
        df.append(path)
    return df
load('/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv'
     '/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv')