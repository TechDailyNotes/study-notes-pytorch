from math import ceil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


# Step 0: Hyperparameter Setup
batch_size = 4
num_epochs = 10


# Step 1: Data Setup
class WineDataset(Dataset):
    def __init__(self, transform=None):
        # Step 1: Data Source
        wine = np.loadtxt(
            './wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1,
        )

        # Step 2: Data Preprocessing

        # Step 3: Data Formatting
        self.X = wine[:, 1:]
        self.y = wine[:, [0]]

        # Step 4: Data Features
        self.num_examples, self.num_features = self.X.shape

        # Step 5: Data Accessories
        self.transform = transform

    def __getitem__(self, index):
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.num_examples


class ToTensor:
    def __call__(self, sample):
        X, y = sample
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y


class MulByFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        X, y = sample
        X *= self.factor
        return X, y


composed = torchvision.transforms.Compose([
    ToTensor(),
    MulByFactor(2),
])


dataset = WineDataset(transform=composed)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

# X, y = next(iter(dataloader))
# print(f"X = {X}")
# print(f"X.shape = {X.shape}")
# print(f"y = {y}")
# print(f"y.shape = {y.shape}")

# print("\n")

# X, y = dataset[0]
# print(f"X = {X}")
# print(f"X.shape = {X.shape}")
# print(f"y = {y}")
# print(f"y.shape = {y.shape}")

# Step 2: Model Setup

# Step 3: Training Loop
num_examples = len(dataset)
num_batches = ceil(num_examples / batch_size)

for epoch in range(num_epochs):
    for batch, (X, y) in enumerate(dataloader):
        if batch % 5 == 0:
            print(f"Epoch {epoch+1} / {num_epochs}, Batch {batch+1} / {num_batches}")  # noqa: E501

# Step 4: Result Test
