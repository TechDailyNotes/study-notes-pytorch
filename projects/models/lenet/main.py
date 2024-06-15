import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1),
            padding=(0, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1),
            padding=(0, 0),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5),
            stride=(1, 1), padding=(0, 0),
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))          # batch_size * 6 * 28 * 28
        out = self.pool(out)                 # batch_size * 6 * 14 * 14
        out = F.relu(self.conv2(out))        # batch_size * 16 * 10 * 10
        out = self.pool(out)                 # batch_size * 16 * 5 * 5
        out = F.relu(self.conv3(out))        # batch_size * 120 * 1 * 1
        out = out.reshape(out.shape[0], -1)  # batch_size * 120
        out = F.relu(self.fc1(out))          # batch_size * 84
        out = self.fc2(out)                  # batch_size * 10
        return out


x = torch.randn((32, 1, 32, 32), dtype=torch.float32).to(device)
model = LeNet().to(device)
out = model(x)
print(f"out.shape = {out.shape}")
