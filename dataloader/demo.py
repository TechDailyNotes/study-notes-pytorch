import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):
    def __init__(self):
        super(WineDataset, self).__init__()

        wine = np.loadtxt(
            './wine.csv',
            delimiter=',',
            dtype=np.float32,
            skiprows=1,
        )

        self.X = torch.from_numpy(wine[:, 1:])  # shape = (178, 13)
        self.y = torch.from_numpy(wine[:, [0]])  # shape = (178, 1)

        # print(f"self.X.shape = {self.X.shape}")
        # print(f"self.y.shape = {self.y.shape}")

        self.n_samples, self.n_features = self.X.shape

        # print(f"self.n_samples = {self.n_samples}")
        # print(f"self.n_features = {self.n_features}")

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
)

print(f"dataloader = {dataloader}")
print(f"dataloader.shape = {len(dataloader)}")
features, labels = next(iter(dataloader))
print(features.size(), labels.size())

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(f"features = \n{features}")
# print(f"labels = \n{labels}")
