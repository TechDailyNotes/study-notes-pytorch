import torch

x = torch.arange(10)
print(f"torch.where(x > 5, x, x * 2) = {torch.where(x > 5, x, x * 2)}")
print(f"x.where(x > 5, x * 2) = {x.where(x > 5, x * 2)}")
