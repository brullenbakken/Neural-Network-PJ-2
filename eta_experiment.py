"""
eta_experiment.py
---
Source code for figure: bar_eta_avg20runs.png
---
This program runs an experiment using the neural network.
Several values of eta are tested, for each eta the number of epochs needed
to correctly classify 80% of the validation data is found. As well as the
percentage of correctly classified validation data after 100 epochs.
The average is taken of the results of 20 experiments (for reliability).
"""
from network import Network

import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def train_network(data, data_v, epochs, mini_batch_size, eta, act_functions):
    """Train the network, and return it. Optional: plot the number of
    correct classifications per epoch."""
    net = Network([len(data[0][0]), 25, 5], act_functions)

    net.SGD(data, epochs, mini_batch_size, eta, data_v)

    return net


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


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

# Experiment.
np.random.seed(0)
random.seed(0)

eta_vals = np.arange(1, 21, 2) / 10
epochs = 100

eps = []
corr = []
for eta in eta_vals:
    eps_temp = []
    corr_temp = []

    # Run the same experiment 20 times for reliablity.
    for _ in range(20):
        while True:
            net = train_network(
                data,
                data_v,
                epochs,
                100,
                eta,
                [[sigmoid, sigmoid], [sigmoid_prime, sigmoid_prime]],
            )
            if len(np.where(np.array(net.correct) >= 200)[0]) > 0:
                break
        eps_temp.append(np.where(np.array(net.correct) >= 200)[0][0])
        corr_temp.append(np.max(net.correct))

    eps.append(eps_temp)
    corr.append(corr_temp)

    print("DONE WITH ETA:", eta)

# Calculate the standard deviations and averages of the experiments (per eta).
eps_err = np.std(eps, axis=1)
corr_err = np.std(corr, axis=1)

eps = np.mean(eps, axis=1)
corr = np.mean(corr, axis=1)

# Plot results.
fig, ax = plt.subplots()
ax2 = ax.twinx()

# Scale correlation results for a neater y axis.
corr = np.array(corr) / 2.5
corr = 2 * (corr / 80) * (corr < 80) + (4 * (corr - 80) / 20 + 2) * (
    corr >= 80
)
corr_err = np.array(corr_err)
corr_err = 2 * (corr_err / 80) * (corr < 80) + 4 * (corr_err / 20) * (
    corr >= 80
)

# Plot to visualise experiment results.
ax.bar(
    [eta - 0.0375 for eta in eta_vals],
    eps,
    0.075,
    yerr=eps_err,
    alpha=0.5,
    color="b",
    error_kw=dict(elinewidth=1.5, markeredgewidth=1.5),
    capsize=2,
)
ax2.bar(
    [eta + 0.0375 for eta in eta_vals],
    np.array(corr),
    0.075,
    yerr=corr_err,
    alpha=0.5,
    color="r",
    error_kw=dict(elinewidth=1.5, markeredgewidth=1.5),
    capsize=2,
)

# Set plot labels.
ax.set_xlabel("eta")
ax.set_ylabel("benodigde epochs voor 80% correcte classificatie", color="b")
ax2.set_ylabel("correct geclassificeerd na 100 epochs (%)", color="r")

# Set plot ticks.
ax.set_xticks(eta_vals)
ax.set_yticks(
    np.linspace(
        0,
        np.int(10 * np.ceil(np.max(eps + np.std(eps)) / 10)),
        np.int(np.ceil(np.max(eps + np.std(eps)) / 10)) + 1,
    )
)
ax2.set_yticks([0, 2, 3, 4, 5, 6])
ax2.set_yticklabels([0, 80, 85, 90, 95, 100])

ax.set_ylim(0, np.int(10 * np.ceil(np.max(eps + np.std(eps)) / 10)))
ax2.set_ylim(0, 6)

plt.title("Effect van de leersnelheid eta (over 250 validatie data)")
plt.show()
