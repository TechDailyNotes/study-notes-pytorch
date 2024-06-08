# import matplotlib.pyplot as plt
# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# Step 1: Hyperparameter Setup
# Type 1.1: Device Params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('runs/practice_ffn_mnist')

# Type 1.2: Data Params
batch_size = 100

# Type 1.3: Model Params
input_size = 784
hidden_size = 100
num_classes = 10
lr = 0.01

# Type 1.4: Training Params
num_epochs = 1

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

train_examples, train_labels = next(iter(train_dataloader))
print(f"train_examples.shape = {train_examples.shape}")
print(f"train_labels.shape = {train_labels.shape}")
print(f"len(train_dataset) = {len(train_dataset)}")
print(f"len(train_dataloader) = {len(train_dataloader)}")
print(f"train_labels[0] = {train_labels[0]}")

print("\n")

test_examples, test_labels = next(iter(test_dataloader))
print(f"test_examples.shape = {test_examples.shape}")
print(f"test_labels.shape = {test_labels.shape}")
print(f"len(test_dataset) = {len(test_dataset)}")
print(f"len(test_dataloader) = {len(test_dataloader)}")
print(f"test_labels[0] = {test_labels[0]}")

print("\n")

# for i in range(100):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(train_examples[i][0], cmap='gray')
# plt.show()

image_grid = make_grid(train_examples)
writer.add_image('mnist_image', image_grid)
# writer.close()
# sys.exit()


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

writer.add_graph(model, train_examples.reshape(-1, input_size))
# writer.close()
# sys.exit()

# Step 4: Training Loop
num_batches = len(train_dataloader)

batch_loss = 0.0
batch_acc = 0

for epoch in range(num_epochs):
    for batch, (examples, labels) in enumerate(train_dataloader):
        examples = examples.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(examples)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        batch_acc += torch.sum(predictions == labels).item() / batch_size

        if batch % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, batch {batch+1}/{num_batches}, loss {loss.item()}")  # noqa: E501

            writer.add_scalar(
                'loss', batch_loss / 100, epoch * num_batches + batch,
            )
            writer.add_scalar(
                'accuracy', batch_acc / 100, epoch * num_batches + batch,
            )
            batch_loss = 0.0
            batch_acc = 0

# Step 5: Result Test
pred_values = []
pred_labels = []

with torch.no_grad():
    num_correct = 0

    for examples, labels in test_dataloader:
        examples = examples.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(examples)
        _, predictions = torch.max(outputs, 1)
        num_correct += (predictions == labels).sum().item()

        outputs_softmax = [F.softmax(output, dim=0) for output in outputs]
        pred_values.append(outputs_softmax)
        pred_labels.append(predictions)

    acc = 100 * num_correct / len(test_dataset)
    print(f"accuracy {acc}%")

    pred_values = torch.cat([torch.stack(batch) for batch in pred_values])
    pred_labels = torch.cat(pred_labels)

    for i in range(10):
        pred_values_i = pred_values[:, i]
        pred_labels_i = pred_labels == i
        writer.add_pr_curve(
            str(i), pred_labels_i, pred_values_i, global_step=0
        )

writer.close()
