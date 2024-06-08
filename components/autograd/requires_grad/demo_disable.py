import torch

x = torch.randn(3, requires_grad=True)
print(x)

x.requires_grad_(False)
print(x)

x.requires_grad_(True)
print(x)

y = x.detach()
print(f"x = {x}")
print(f"y = {y}")

with torch.no_grad():
    z = x + 2
    print(f"x = {x}")
    print(f"z = {z}")
