import torch

print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.version.cuda = {torch.version.cuda}")
