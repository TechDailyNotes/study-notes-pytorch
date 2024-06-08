import torch

w = torch.ones(4, requires_grad=True)

for epoch in range(3):
    o = (w * 3).sum()
    o.backward()
    print(f"w.grad = {w.grad}")
    w.grad.zero_()

# optimizer = torch.optim.SGD(w, lr=0.001)
# optimizer.step()
# optimizer.zero_grad()
