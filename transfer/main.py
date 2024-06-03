from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import make_grid


# Step 1: Hyperparameter Setup
phases = ['train', 'val']

# Step 1.1: Device Params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1.2: Data Params
batch_size = 4

# Step 1.3: Model Params
lr = 0.001

# Step 1.4: Training Params
num_epochs = 1

# Step 1.5: Param Tuning

# Step 2: Data Setup
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Step 2.1: Data Preprocessing
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

# Step 2.2: Data Source
folder = 'data/hymenoptera_data'

image_datasets = {
    phase: ImageFolder(
        os.path.join(folder, phase),
        transform=image_transforms[phase]
    ) for phase in phases
}

# Step 2.3: Data Formatting
image_dataloaders = {
    phase: DataLoader(
        dataset=image_datasets[phase],
        batch_size=batch_size,
        shuffle=(phase == 'train')
    ) for phase in phases
}

# Step 2.4: Data Features
classes = image_datasets['train'].classes
dataset_sizes = {phase: len(image_datasets[phase]) for phase in phases}


def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


print()

print(f"classes = {classes}")

print()

train_images, train_labels = next(iter(image_dataloaders['train']))
print(f"len(image_datasets['train']) = {len(image_datasets['train'])}")
print(f"len(image_dataloaders['train']) = {len(image_dataloaders['train'])}")
print(f"train_images.shape = {train_images.shape}")
print(f"train_labels.shape = {train_labels.shape}")
print(f"train_labels = {train_labels}")
inp = make_grid(train_images)
# imshow(inp, title=[classes[label] for label in train_labels])

print()

val_images, val_labels = next(iter(image_dataloaders['val']))
print(f"len(image_datasets['val']) = {len(image_datasets['val'])}")
print(f"len(image_dataloaders['val']) = {len(image_dataloaders['val'])}")
print(f"val_images.shape = {val_images.shape}")
print(f"val_labels.shape = {val_labels.shape}")
print(f"val_labels = {val_labels}")
inp = make_grid(val_images)
# imshow(inp, title=[classes[label] for label in val_labels])

print()

# Step 2.5: Data Accessories

print("Data loading completed!\n")

# Step 3: Model Setup
# Step 3.1: Architecture Setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

# Step 3.2: Loss Function Setup
criterion = nn.CrossEntropyLoss()

# Step 3.3: Optimizer Setup
optimizer = optim.SGD(model.parameters(), lr=lr)

# Step 3.4: Scheduler Setup
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Step 4: Training Loop
def train_model(model, criterion, optimizer, scheduler):
    best_acc = 0.0
    best_wts = deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        for phase in phases:
            time_start = time()

            epoch_loss = 0.0
            epoch_acc = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for _, (images, labels) in enumerate(image_dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, predictions = torch.max(outputs, 1)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss.item() * images.shape[0]
                    epoch_acc += torch.sum(predictions == labels).item()

            if phase == 'train':
                scheduler.step()

            time_end = time()
            time_taken = time_end - time_start

            epoch_loss /= dataset_sizes[phase]
            epoch_acc /= dataset_sizes[phase]

            print(f"epoch {epoch + 1}/{num_epochs}, phase {phase}, loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}, time {time_taken:.4f}s")  # noqa: E501

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model


model = train_model(model, criterion, optimizer, scheduler)

print("Model training completed!\n")

# Step 5: Result Test
