import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

n_epochs = 10000
lr = 0.01

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
input_dim = n_features
output_dim = 1

# print(f"type(X) = {type(X)}")
# print(f"X.shape = {X.shape}")
# print(f"n_samples = {n_samples}, n_features = {n_features}")

# print(f"type(Y) = {type(y)}")
# print(f"type(Y[0]) = {type(y[0])}")
# print(f"Y.shape = {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


class LogitsticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogitsticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        X = self.linear(X)
        y_pred = torch.sigmoid(X)
        return y_pred


model = LogitsticRegression(input_dim, output_dim)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss}")

with torch.no_grad():
    y_pred = model(X_test).round()
    acc = y_pred.eq(y_test).sum() / y_test.shape[0]

    print(f"Accuracy is {acc}")
