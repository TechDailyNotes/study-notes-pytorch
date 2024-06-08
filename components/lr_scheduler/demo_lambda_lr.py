import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


lr = 0.01

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch / 10)

print(f"scheduler.state_dict()['base_lrs'][0] = {scheduler.state_dict()['base_lrs'][0]}")  # noqa: E501

for epoch in range(10):
    optimizer.step()
    scheduler.step()
    print(f"optimizer.state_dict()['param_groups'][0]['lr'] = {optimizer.state_dict()['param_groups'][0]['lr']}")  # noqa: E501
