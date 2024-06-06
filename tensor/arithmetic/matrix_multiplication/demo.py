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

x2 = torch.ones(3, 2, 3)
y2 = torch.ones(3, 3, 4)
print(f"x2 = {x2}")
print(f"y2 = {y2}")

z2_1 = torch.bmm(x2, y2)
z2_2 = x2.bmm(y2)
print(f"z2_1 = {z2_1}")
print(f"x2_2 = {z2_2}")
