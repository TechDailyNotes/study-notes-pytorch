import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0


def forward(X):
    return w * X


def loss(Y, Y_):
    return ((Y_ - Y) ** 2).mean()


def gradient(X, Y, Y_):
    return np.dot(2 * (Y_ - Y), X).mean()
