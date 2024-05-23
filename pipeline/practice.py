import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

n_iters = 80
lr = 0.01


def forward(X):
    return w * X


loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=lr)

print(f"Prediction before training is {forward(5):.6f}")

for epoch in range(n_iters):
    Y_pred = forward(X)
    loss_ = loss(Y, Y_pred)
    loss_.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss {loss_:.10f}")

print(f"Prediction after training is {forward(5):.6f}")
