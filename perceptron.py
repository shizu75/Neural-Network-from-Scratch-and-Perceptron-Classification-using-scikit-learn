import numpy as np
import pandas as pd

data = pd.read_csv(r"D:\Internship\Pandas\pre.csv")
X = data.iloc[:,:-1].values
Y = data.iloc[:, -1].values
Y = Y.reshape(Y.size, 1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigder(x):
    return x * (1 - x)

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Synaptic weights: ", synaptic_weights)

for iterations in range(100000):
    input_layer = X
    output = sigmoid(np.dot(X, synaptic_weights))
    error = output - Y
    adjustments = error * sigder(Y)
    synaptic_weights += np.dot(X.T, adjustments)

print("Synaptic Weights: " , synaptic_weights)
print("Output after weights: ", output)
