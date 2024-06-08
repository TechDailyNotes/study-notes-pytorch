# import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

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

        # Step 1.5: Data Accessories
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


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        X, y = sample
        X *= self.factor
        return X, y


dataset1 = WineDataset(transform=ToTensor())
dataloader1 = DataLoader(
    dataset=dataset1,
    batch_size=batch_size,
    shuffle=True,
)

X1, y1 = next(iter(dataloader1))
print(f"X1 = {X1}")
print(f"y1 = {y1}")

composed = torchvision.transforms.Compose([
    ToTensor(),
    MulTransform(2),
])
dataset2 = WineDataset(transform=composed)
dataloader2 = DataLoader(
    dataset=dataset2,
    batch_size=batch_size,
    shuffle=True,
)

X2, y2 = next(iter(dataloader2))
print(f"X2 = {X2}")
print(f"y2 = {y2}")

# Step 2: Model Setup

# Step 3: Training Loop
# n_examples = len(dataset)
# n_batches = math.ceil(n_examples / batch_size)

# for epoch in range(n_epochs):
#     for batch, (X, y) in enumerate(dataloader):
#         if batch % 5 == 0:
#             print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch+1}/{n_batches}, X.shape = {X.shape}, y.shape = {y.shape}")  # noqa: E501

# Step 4: Result Test
