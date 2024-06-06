import torch

x = torch.rand(5)
x_clamp = x.clamp(min=0.4, max=0.6)
print(f"x = {x}")
print(f"x_clamp = {x_clamp}")
z = torch.clamp(x, min=0.4, max=0.6)
print(f"z = {z}")
