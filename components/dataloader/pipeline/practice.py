import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Step 0: Hyperparameter Setup
n_epochs = 2
batch_size = 4


# Step 1: Data Setup
class WineDataset(Dataset):
    def __init__(self):
        # Step 1.1: Data Source
        wine = np.loadtxt(
            './wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1,
        )

        # Step 1.2: Data Preprocessing

        # Step 1.3: Data Formatting
        self.X = torch.from_numpy(wine[:, 1:])
        self.y = torch.from_numpy(wine[:, [0]])

        # Step 1.4: Data Features
        self.n_examples, self.n_features = self.X.shape

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_examples


dataset = WineDataset()
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Step 2: Model Setup

# Step 3: Training Loop
for epoch in range(n_epochs):
    for batch, (X, y) in enumerate(dataloader):
        if batch % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch+1}/{batch_size}, X.shape = {X.shape}, y.shape = {y.shape}")  # noqa: E501

# Step 4: Result Test
