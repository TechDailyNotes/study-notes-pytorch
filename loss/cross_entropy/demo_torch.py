import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

y = torch.tensor([0], dtype=torch.int64)
y_pred_good = torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32)
y_pred_bad = torch.tensor([[1.0, 2.0, 5.0]], dtype=torch.float32)

print(f"Good prediction cross entropy is {loss(y_pred_good, y).item()}")
print(f"Bad prediction cross entropy is {loss(y_pred_bad, y).item()}")

value1, label1 = torch.max(y_pred_good, 1)
value2, label2 = torch.max(y_pred_bad, 1)

print(f"value1 = {value1}, label1 = {label1}")
print(f"value2 = {value2}, label2 = {label2}")
