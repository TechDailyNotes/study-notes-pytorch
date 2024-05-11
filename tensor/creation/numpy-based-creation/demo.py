import numpy as np
import torch

x = torch.ones(5, dtype=torch.int64)
print(f"x = {x}")
y = x.numpy()
print(f"y = {y}, type(y) = {type(y)}")
x.add_(1)
print(f"x = {x}")
print(f"y = {y}")

a = np.ones(5)
print(f"a = {a}")
b = torch.from_numpy(a)
print(f"b = {b}")
a += 1
print(f"a = {a}")
print(f"b = {b}")
