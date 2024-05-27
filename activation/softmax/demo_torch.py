import torch

x = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float32)
x = torch.softmax(x, dim=0)
print(f"x = {x}")
