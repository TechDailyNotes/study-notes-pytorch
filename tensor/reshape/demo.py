import torch

x = torch.arange(9)
x_view = x.view(3, 3)
x_reshape = x.reshape(3, 3)
print(x_view.shape)
print(x_reshape.shape)
print(x.is_contiguous())

y = x_view.t()
print(y.is_contiguous())
y_view = y.contiguous().view(9)
print(y_view)
