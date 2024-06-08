import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0


def forward(X):
    return w * X


def loss(Y, Y_hat):
    return ((Y_hat - Y) ** 2).mean()


def gradient(X, Y, Y_hat):
    return np.dot(2 * (Y_hat - Y), X).mean()


n_iters = 20
lr = 0.01

print(f"Prediction before training is {forward(5):.6f}")

for epoch in range(n_iters):
    Y_hat = forward(X)
    loss_ = loss(Y, Y_hat)

    dw = gradient(X, Y, Y_hat)
    w -= lr * dw

    print(f"Epoch {epoch + 1}, Loss {loss_:.10f}")

print(f"Prediction after training is {forward(5):.6f}")
