import torch
import torch.nn as nn


class MulNeuNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MulNeuNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


mul_model = MulNeuNet(input_size=28*28, hidden_size=5, output_size=3)
ce_loss = nn.CrossEntropyLoss()


class BinNeuNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinNeuNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out


bin_model = BinNeuNet(input_size=28*28, hidden_size=5)
bce_loss = nn.BCELoss()
