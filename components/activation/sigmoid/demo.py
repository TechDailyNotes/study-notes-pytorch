import torch
import torch.nn as nn
import torch.nn.functional as F


class NN1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()
        # self.tanh = nn.Tanh()
        # self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class NN2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # torch.tanh()
        # torch.softmax()
        return out


class NN3(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN3, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.sigmoid(self.linear2(out))
        # F.leaky_relu
        # F.tanh()
        # F.softmax()
        return out
