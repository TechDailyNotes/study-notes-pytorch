import torch

x = torch.rand(4, 4, dtype=torch.float16)
print(f"x = {x}, {x.shape}, {x.size()}")
y = x.view(16)
print(f"y = {y}, {y.shape}, {y.size()}")
z = x.view(8, -1)
print(f"z = {z}, {z.shape}, {z.size()}")
