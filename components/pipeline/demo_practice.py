import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
input_dim = n_features
output_dim = n_features

X_test = torch.tensor([5], dtype=torch.float32)

n_iters = 100
lr = 0.01


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


model = LinearRegression(input_dim, output_dim)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

print(f"Prediction before training is {model(X_test).item():.6f}")

for epoch in range(n_iters):
    Y_pred = model(X)
    loss = loss_fn(Y, Y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss {loss:.10f}")

print(f"Prediction after training is {model(X_test).item():.6f}")
