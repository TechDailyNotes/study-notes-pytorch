import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
x = softmax(x)
print(f"x = {x}")
print(f"np.sum(x) = {np.sum(x, axis=0)}")
