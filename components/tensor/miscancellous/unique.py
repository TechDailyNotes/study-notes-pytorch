import torch

x = torch.tensor([0, 0, 1, 1, 1, 2]).unique()
print(f"x = {x}")

print(f"torch.unique(torch.tensor([0, 0, 1])) = {torch.unique(torch.tensor([0, 0, 1]))}")  # noqa: E501
