import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

# Step 0: Hyperparameter Setup
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
        self.transform = transform

        # Step 1.3: Data Formatting
        self.X = wine[:, 1:]
        self.y = wine[:, [0]]

        # Step 1.4: Data Features
        self.n_samples, self.n_features = self.X.shape

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        X, y = sample
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y


class ScaleByFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        X, y = sample
        X *= self.factor
        return X, y


composed = torchvision.transforms.Compose([
    ToTensor(),
    ScaleByFactor(2),
])

dataset = WineDataset(transform=composed)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

X, y = next(iter(dataloader))

print(f"X = {X}")
print(f"type(X) = {type(X)}")
print(f"X.shape = {X.shape}")

print(f"y = {y}")
print(f"type(y) = {type(y)}")
print(f"y.shape = {y.shape}")

# Step 2: Model Setup

# Step 2.1: Architecture Setup

# Step 2.2: Loss Setup

# Step 2.3: Optimizer Setup

# Step 3: Training Loop

# Step 4: Result Test
