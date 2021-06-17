"""
validationgenerator.py
---
Source code for validation.csv
---
This file creates pairs of input and output data for the neural network.
Each pair consists of a list of numbers generated by some
distribution (input data), and a digit indicating which distribution
was used (desired neural network output).
Note that the validation data is different from the distribution data used
to train the network to prevent overlearning (the network only correctly
classifying the given training data).
"""
import numpy as np
import pandas as pd


data = []

n = 50

# Exponential distribution.
for _ in range(n):
    values = list(np.random.exponential(np.random.random(), 50))
    values.sort()
    values.append(0)
    data.append(values)

# Lognormal distribution.
for _ in range(n):
    a = np.random.randn()
    values = list(np.random.lognormal(a, np.random.random(), 50))
    values.sort()
    values.append(1)
    data.append(values)

# Gamma distribution.
for _ in range(n):
    values = list(np.random.gamma(2, np.random.random(), 50))
    values.sort()
    values.append(2)
    data.append(values)

# Uniform distribution.
for _ in range(n):
    a = np.random.randn()
    b = np.random.randn()
    while b < a:
        b = np.random.randn()
    values = list((b - a) * np.random.random(50) + a)
    values.sort()
    values.append(3)
    data.append(values)

# Normal distribution.
for _ in range(n):
    a = np.random.randn()
    values = list(np.random.normal(a, np.random.random(), 50))
    values.sort()
    values.append(4)
    data.append(values)

np.random.shuffle(data)

Output = pd.DataFrame(data,)
Output.to_csv("validation.csv", sep="\t", encoding="utf-8", index=False)
