"""
correct_per_epoch.py
---
This program runs the neural network as a test.
After all epochs (training phases in which gradient descent is performed) the
number of correctly classified validation data per epoch is plotted.
"""
from network import Network

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


# Activation functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def leakyrelu(z):
    """The Leaky ReLU function."""
    return np.maximum(0.1 * z, z)


def leakyrelu_prime(z):
    """Derivative of the Leaky ReLu function."""
    return 0.1 * (z < 0) + (z >= 0)


def train_network(data, data_v, epochs, mini_batch_size, eta, act_functions):
    """Train the network, and return it. Optional: plot the number of
    correct classifications per epoch."""
    net = Network([len(data[0][0]), 25, 5], act_functions)

    net.SGD(data, epochs, mini_batch_size, eta, data_v)

    return net


# Read data and start network.
df = pd.read_csv("distributions.csv", delimiter="\t")
data_origineel = df.to_records(index=False)

data = []
for row in data_origineel:
    row1 = np.asarray([*row])
    inp = row1[:-1]
    outp = row1[-1]
    data.append((np.transpose([inp]), outp))

df = pd.read_csv("validation.csv", delimiter="\t")
data_origineel_v = df.to_records(index=False)

data_v = []
for row in data_origineel_v:
    row1 = np.asarray([*row])
    inp = row1[:-1]
    outp = row1[-1]
    data_v.append((np.transpose([inp]), outp))

# Plot the number of correctly classified validation data per epoch.
epochs = 1000

net = train_network(
    data,
    data_v,
    epochs,
    100,
    0.5,
    [[sigmoid, sigmoid], [sigmoid_prime, sigmoid_prime]],
)

plt.scatter(range(1, epochs + 1), net.correct, label=net.act[0])
plt.xlabel("epochs")
plt.ylabel("correct ({} validatie data)".format(len(data_v)))

plt.axis([None, None, 0, len(data_v)])

plt.title("Aantal correcte output per epoch (validatie data)")
plt.legend()
plt.show()
