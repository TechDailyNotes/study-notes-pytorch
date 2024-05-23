import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(X):
    return w * X


def loss(X, Y, Y_hat):
    return ((Y_hat - Y) ** 2).mean()


n_iters = 20
lr = 0.01

print(f"Prediction before training is {forward(5):.6f}")

for epoch in range(n_iters):
    Y_hat = forward(X)
    loss_ = loss(X, Y, Y_hat)
    loss_.backward()

    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()

    print(f"Epoch {epoch + 1}, Loss {loss_:.10f}")

print(f"Prediction after training is {forward(5):.6f}")
