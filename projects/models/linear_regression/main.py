import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim

X_np, Y_np = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=1,
)

X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape
input_dim = n_features
output_dim = 1

n_epochs = 1000
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

for epoch in range(n_epochs):
    Y_pred = model(X)
    loss = loss_fn(Y, Y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

Y_pred = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, Y_pred, 'b')
plt.show()
