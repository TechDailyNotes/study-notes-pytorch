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


print(f"Prediction before training is {forward(5):.3f}")

n_iters = 20
lr = 0.01

for epoch in range(n_iters):
    Y_ = forward(X)
    los = loss(Y, Y_)

    dw = gradient(X, Y, Y_)
    w -= lr * dw

    print(f"Epoch {epoch + 1}, Loss {los:.6f}")

print(f"Prediction after training is {forward(5):.3f}")
