import torch

x = torch.rand(5, 3, dtype=torch.double)
print(f"x = {x}")
print(f"x[:, 0] = {x[:, 0]}")
print(f"x[0, :] = {x[0, :]}")
print(f"x[1, 1] = {x[1, 1].item()}")
