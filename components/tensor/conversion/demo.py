import torch

x = torch.arange(4, dtype=torch.float32)
print(f"x = {x}")
print(f"x.bool() = {x.bool()}")
print(f"x.half() = {x.half()}")
print(f"x.short() = {x.short()}")
print(f"x.long() = {x.long()}")
