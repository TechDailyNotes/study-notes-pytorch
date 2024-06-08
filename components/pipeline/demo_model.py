import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

n_iters = 100
lr = 0.01


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


model = LinearRegression(input_size, output_size)


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

print(f"Prediction before training is {model(X_test).item():.6f}")

for epoch in range(n_iters):
    Y_pred = model(X)
    loss_ = loss(Y, Y_pred)
    loss_.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}, Loss {loss_:.10f}")

print(f"Prediction after training is {model(X_test).item():.6f}")
