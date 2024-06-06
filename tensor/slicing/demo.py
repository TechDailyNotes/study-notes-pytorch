import torch

x = torch.rand(5, 3, dtype=torch.double)
print(f"x = {x}")
print(f"x[:, 0] = {x[:, 0]}")
print(f"x[0, :] = {x[0, :]}")
print(f"x[1, 1] = {x[1, 1].item()}")

index = [0, 1, 4]
print(f"x[index, 0] = {x[index, 0]}")

y = torch.rand(5, 5)
print(f"y = {y}")
rows = [0, 2, 4]
cols = [1, 3, 4]
print(f"y[rows][cols] = {y[rows, cols]}")

z = torch.arange(10)
print(f"z = {z}")
print(f"z[(z < 3) | (z > 7)] = {z[(z < 3) | (z > 7)]}")
print(f"z[z.remainder(2) == 0] = {z[z % 2 == 0]}")
