import torch

x = torch.ones(5)
y = torch.ones(5)
print(f"x = {x}")
print(f"y = {y}")

z1 = x * y
z2 = x.dot(y)
z3 = x * 1
print(f"z1 = {z1}")
print(f"z2 = {z2.item()}")
print(f"type(z2) = {type(z2)}")
print(f"z3 = {z3}")
