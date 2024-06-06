import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x1 = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.int64, device=device)
print(x1, x1.dtype, x1.size())
print(x1.device)
