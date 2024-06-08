import torch

x = torch.ones(5, 5)
x_pow = x.matrix_power(2)
print(f"x = {x}")
print(f"x_pow = {x_pow}")

y = torch.ones(5, 5)
y_pow = torch.matrix_power(y, 2)
print(f"y = {y}")
print(f"y_pow = {y_pow}")
