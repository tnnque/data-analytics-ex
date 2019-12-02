#The Data
import numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features= 1, noise= 0.4, bias= 50)

#Hypothesis Function
def hypothesis(theta0, theta1, x):
    return theta0 + (theta1 * x)

def plotline(theta0, theta1, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100
