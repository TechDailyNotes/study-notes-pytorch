import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

y_hat = x * w
loss = (y - y_hat) ** 2
print(f"loss = {loss}")
print(f"type(loss) = {type(loss)}")
print(f"loss.grad_fn = {loss.grad_fn}")

loss.backward()
print(f"w.grad = {w.grad}")
