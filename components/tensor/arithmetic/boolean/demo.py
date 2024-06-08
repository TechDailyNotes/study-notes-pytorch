import torch

x = torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)
print(f"x = {x}")
y = x.any()
print(f"y = {y.item()}")
print(f"type(y) = {type(y.item())}")
z = x.all()
print(f"z = {z.item()}")
print(f"type(z) = {type(z.item())}")
