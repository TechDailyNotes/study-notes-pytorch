import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x1 = torch.empty(2, 2, 2, 2)
print(x1)

x2 = torch.rand(2, 2, 2, 2)
print(x2)

x3 = torch.zeros(2, 2, 2, 2)
print(x3)

x4 = torch.ones(2, 2, 2, 2)
print(x4, x4.dtype)

x5 = torch.ones(2, 2, 2, 2, dtype=torch.double)
print(x5, x5.dtype)

x6 = torch.zeros(2, 2, 2, 2, dtype=torch.float16)
print(x6, x6.dtype)

x7 = torch.empty(2, 2, 2, 2, dtype=torch.int)
print(x7, x7.dtype, x7.size())

x8 = torch.eye(5, 5, dtype=torch.float32)
print(x8, x8.shape)

x9 = torch.arange(0, 5, 1)
print(x9, x9.shape)

x10 = torch.linspace(0.1, 1, 10, dtype=torch.float32, device=device)
print(x10)

x11 = torch.empty(1, 5, dtype=torch.float32).normal_(mean=0, std=1)
print(f"x11 = {x11}")

x12 = torch.empty(1, 5, dtype=torch.float32, device=device).uniform_(0, 1)
print(f"x12 = {x12}")

x13 = torch.diag(torch.ones(3, dtype=torch.float32, device=device))
print(f"x13 = {x13}")
