import torch

x1 = torch.rand(2, 2)
print(f"x1 = {x1}")
y1 = torch.rand(2, 2)
print(f"y1 = {y1}")
z1 = x1 + y1
print(f"z1 = {z1}")
z2 = torch.add(x1, y1)
print(f"z2 = {z2}")
y1.add_(x1)
print(f"x1 = {x1}")
print(f"y1 = {y1}")

x3 = torch.rand(2, 2, dtype=torch.float16)
y3 = torch.rand(2, 2, dtype=torch.float16)
print(f"x3 = {x3}")
print(f"y3 = {y3}")
z3 = x3 - y3
print(f"z3 = {z3}")
z4 = torch.sub(x3, y3)
print(f"z4 = {z4}")
y3.sub_(x3)
print(f"y3 = {y3}")
