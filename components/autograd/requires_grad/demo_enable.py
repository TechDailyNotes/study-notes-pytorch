import torch

x = torch.randn(3, requires_grad=True)
print(f"x = {x}")

y = x + 2
print(f"y = {y}")

z = y ** 2 * 2
print(f"z = {z}")

z = z.mean()
z.backward()

print(f"x.grad = {x.grad}")

x1 = torch.randn(3, requires_grad=True)
y1 = x1 + 2
z1 = y1 ** 2 * 2
v1 = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
z1.backward(v1)
print(f"x1.grad = {x1.grad}")
