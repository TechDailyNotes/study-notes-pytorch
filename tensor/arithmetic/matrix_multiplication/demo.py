import torch

x = torch.rand(2, 3)
y = torch.rand(3, 5)

z1 = torch.mm(x, y)
print(f"x = {x}")
print(f"y = {y}")
print(f"z1 = {z1}")

z2 = x.mm(y)
print(f"z2 = {z2}")

z3 = x @ y
print(f"z3 = {z3}")

# x @= y
# print(f"x = {x}")
# print(f"y = {y}")

z4 = x.matmul(y)
print(f"x = {x}")
print(f"y = {y}")
print(f"z4 = {z4}")
