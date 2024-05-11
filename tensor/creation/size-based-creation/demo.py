import torch

x1 = torch.empty(2, 2, 2, 2)
print(x1)

x2 = torch.rand(2, 2, 2, 2)
print(x2)

x3 = torch.zeros(2, 2, 2, 2)
print(x3)

x4 = torch.ones(2, 2, 2, 2)
print(x4, x4.dtype)

x5 = torch.ones(2, 2, 2, 2, dtype=torch.double)
print(x5, x5.dtype)

x6 = torch.zeros(2, 2, 2, 2, dtype=torch.float16)
print(x6, x6.dtype)

x7 = torch.empty(2, 2, 2, 2, dtype=torch.int)
print(x7, x7.dtype, x7.size())
