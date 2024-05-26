import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Step 0: Hyperparameter Setup
n_epochs = 2
batch_size = 4


# Step 1: Data Setup
class WineDataset(Dataset):
    def __init__(self, transform=None):
        # Step 1.1: Data Source
        wine = np.loadtxt(
            './wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1,
        )

        # Step 1.2: Data Preprocessing

        # Step 1.3: Data Formatting
        self.X = wine[:, 1:]
        self.y = wine[:, [0]]

        # Step 1.4: Data Features
        self.n_samples, self.n_features = self.X.shape

        self.transform = transform

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        X, y = sample
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y


dataset = WineDataset(transform=ToTensor())
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Step 2: Model Setup

# Step 3: Training Loop
n_examples = len(dataset)
n_batches = math.ceil(n_examples / batch_size)

for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(dataloader):
        if batch % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch+1}/{n_batches}, X.shape = {X.shape}, y.shape = {y.shape}")  # noqa: E501

# Step 4: Result Test
