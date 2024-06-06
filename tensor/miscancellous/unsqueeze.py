import torch

x = torch.rand(10)
print(x.shape)
x.unsqueeze_(0)
print(x.shape)
x.unsqueeze_(2)
print(x.shape)
x.squeeze_(0)
print(x.shape)
