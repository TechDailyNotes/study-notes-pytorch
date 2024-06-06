import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
seq_len = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
lr = 0.001
num_epochs = 1

train_dataset = MNIST(
    root='./data', train=True, transform=ToTensor(), download=True,
)
test_dataset = MNIST(
    root='./data', train=False, transform=ToTensor(), download=False,
)
train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False,
)


def imshow(image, label):
    plt.imshow(image[0], cmap='gray')
    plt.title(label)
    plt.show()


print()

train_images, train_labels = next(iter(train_dataloader))
num_batches = len(train_dataloader)
print(f"len(train_dataset) = {len(train_dataset)}")
print(f"batch_size = {batch_size}")
print(f"len(train_dataloader) = {len(train_dataloader)}")
print(f"train_images.shape = {train_images.shape}")
print(f"train_labels.shape = {train_labels.shape}")
print(f"train_dataset.classes = {train_dataset.classes}")
print(f"train_labels[0] = {train_labels[0].item()}")
# imshow(train_images[0], train_labels[0].item())

print()

test_images, test_labels = next(iter(test_dataloader))
print(f"len(test_dataset) = {len(test_dataset)}")
print(f"batch_size = {batch_size}")
print(f"len(test_dataloader) = {len(test_dataloader)}")
print(f"test_images.shape = {test_images.shape}")
print(f"test_labels.shape = {test_labels.shape}")
print(f"test_dataset.classes = {test_dataset.classes}")
print(f"test_labels[0] = {test_labels[0].item()}")
# imshow(test_images[0], test_labels[0].item())

print()

print("Data Loading Completed!\n")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def forward(self, x):
        h0 = torch.zeros(
            (self.num_layers, x.size(0), self.hidden_size),
            dtype=torch.float32,
        ).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

batch_loss = 0.0
batch_acc = 0

for epoch in range(num_epochs):
    for batch, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, seq_len, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss += loss.item()
        batch_acc += torch.sum(predictions == labels).item() / batch_size

        if (batch + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, batch {batch + 1}/{num_batches}, loss {batch_loss / 100:.4f}, accuracy {batch_acc / 100 * 100:.4f}%")  # noqa: E501
            batch_loss = 0.0
            batch_acc = 0

print("Model Training Completed!\n")
