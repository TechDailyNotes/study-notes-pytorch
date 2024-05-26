import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):
    def __init__(self):
        # Step 1: Data Source
        wine = np.loadtxt(
            './wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1,
        )

        # Step 2: Data Preprocessing

        # Step 3: Data Formatting
        self.X = torch.from_numpy(wine[:, 1:])
        self.y = torch.from_numpy(wine[:, [0]])

        # Step 4: Data Features
        self.n_examples, self.n_features = self.X.shape

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_examples


dataset = WineDataset()
# print(f"len(dataset) = {len(dataset)}")

# X, y = dataset[0]
# print(f"X.shape[0] = {X.shape[0]}")
# print(f"dataset.n_features = {dataset.n_features}")
# print(f"type(X) = {type(X)}")

# print(f"X = {X}")
# print(f"X.shape = {X.shape}\n")

# print(f"y = {y}")
# print(f"y.shape = {y.shape}\n")

dataloader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
)
# data = next(iter(dataloader))
# X, y = data

# print(f"X = {X}")
# print(f"X[0][0] = {X[0][0].item()}")
# print(f"type(X) = {type(X)}")
# print(f"type(X[0]) = {type(X[0])}")
# print(f"type(X[0][0]) = {type(X[0][0])}")
# print(f"X.shape = {X.shape}\n")

# print(f"y = {y}")
# print(f"y[0][0] = {y[0][0].item()}")
# print(f"type(y) = {type(y)}")
# print(f"type(y[0]) = {type(y[0])}")
# print(f"type(y[0][0]) = {type(y[0][0])}")
# print(f"y.shape = {y.shape}\n")

n_epochs = 2
n_examples = len(dataset)
n_batches = math.ceil(n_examples / 8)
# print(f"n_batches = {n_batches}")

for epoch in range(n_epochs):
    for i, (X, y) in enumerate(dataloader):
        if i % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Step {i + 1}/{n_batches}, X.shape = {X.shape}")  # noqa: E501
