import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data
path = input("Enter path:")
def load(path):
    df = pd.read_csv(path)
    return df

def hypothesis(theta0, theta1, x):
    return theta0 + (theta1 * x)

def plotline(theta0, theta1, X, y):
    max_x = np.max(X) +
    min_x = np.min(X) -

def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0
    for