# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Step 1: Hyperparameter Setup
# Type 1.1: Device Params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Type 1.2: Data Params
batch_size = 100

# Type 1.3: Model Params
input_size = 784  # 28 * 28
hidden_size = 100
num_classes = 10
lr = 0.001

# Type 1.4: Training Params
num_epochs = 2

# Step 2: Data Setup
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True,
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor(),
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
)

train_example, train_label = next(iter(train_dataloader))
print(f"train_example.shape = {train_example.shape}")
print(f"train_label.shape = {train_label.shape}")

test_example, test_label = next(iter(test_dataloader))
print(f"test_example.shape = {test_example.shape}")
print(f"test_label.shape = {test_label.shape}")

print(f"len(train_dataset) = {len(train_dataset)}")
print(f"len(test_dataset) = {len(test_dataset)}")
print(f"len(train_dataloader) = {len(train_dataloader)}")
print(f"len(test_dataloader) = {len(test_dataloader)}")

# for i in range(100):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(train_example[i][0], cmap='gray')
# plt.show()


# Step 3: Model Setup
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out


model = FFN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Step 4: Training Loop
for epoch in range(num_epochs):
    for batch, (examples, labels) in enumerate(train_dataloader):
        examples = examples.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(examples)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, batch {batch+1}/{len(train_dataloader)}, loss {loss.item()}")  # noqa: E501

# Step 5: Result Plot
with torch.no_grad():
    num_correct = 0

    for examples, labels in test_dataloader:
        examples = examples.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(examples)
        _, predictions = torch.max(outputs, 1)
        num_correct += (predictions == labels).sum().item()

    acc = 100 * num_correct / len(test_dataset)
    print(f"acc = {acc}%")
