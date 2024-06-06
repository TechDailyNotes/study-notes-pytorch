import torch

x = torch.rand((2, 5))
y = torch.rand((2, 5))
z1 = torch.cat((x, y), dim=0)
z2 = torch.cat((x, y), dim=1)
print(z1.shape)
print(z2.shape)
