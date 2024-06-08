import torch

x = torch.linspace(-1, 1, 10)
y = x > 0
print(f"x = {x}")
print(f"y = {y}")

x1 = torch.empty(2, 2).normal_(mean=0, std=1)
y1 = x1 > 0
print(f"x1 = {x1}")
print(f"y1 = {y1}")
