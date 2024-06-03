import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Step 1: Hyperparameter Setup
# Step 1.1: Device Params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1.2: Data Params
batch_size = 100

# Step 1.3: Model Params
lr = 0.05

# Step 1.4: Training Params
num_epochs = 3

# Step 1.5: Param Tuning
# lr=0.01, num_epochs=3, accuracy=28.26%
# lr=0.05, num_epochs=5, accuracy=47.98%

# Step 2: Data Setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True,
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, transform=transform, download=False,
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
)

print("Data loaded!\n")

train_examples, train_labels = next(iter(train_dataloader))
print(f"len(train_dataset) = {len(train_dataset)}")
print(f"len(train_dataloader) = {len(train_dataloader)}")
print(f"train_examples.shape = {train_examples.shape}")
print(f"train_labels.shape = {train_labels.shape}")
print(f"train_labels[0].item() = {train_labels[0].item()}")

print("")

test_examples, test_labels = next(iter(test_dataloader))
print(f"len(test_dataset) = {len(test_dataset)}")
print(f"len(test_dataloader) = {len(test_dataloader)}")
print(f"test_examples.shape = {test_examples.shape}")
print(f"test_labels.shape = {test_labels.shape}")
print(f"test_labels[0].item() = {test_labels[0].item()}")

print("")


# Step 3: Model Setup
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 16*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Step 4: Training Loop
num_batches = len(train_dataloader)

for epoch in range(num_epochs):
    for batch, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, batch {batch+1}/{num_batches}, loss {loss.item()}")  # noqa: E501

print("Training loop completed!\n")

# Step 5: Result Test
with torch.no_grad():
    num_tests = len(test_dataset)
    num_correct = 0
    num_label_tests = [0] * 10
    num_label_correct = [0] * 10

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        num_correct += (predictions == labels).sum().item()

        for i in range(min(batch_size, labels.shape[0])):
            label = labels[i]
            prediction = predictions[i]

            num_label_tests[label] += 1
            if label == prediction:
                num_label_correct[label] += 1

    acc = 100 * num_correct / num_tests
    print(f"accuracy {acc}")

    for i in range(10):
        acc = 100 * num_label_correct[i] / num_label_tests[i]
        print(f"label {i}, accuracy {acc}")
