from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
# from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder = 'data/hymenoptera_data'
phases = ['train', 'val']

batch_size = 4
num_epochs = 1

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

image_transforms = {
    phases[0]: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    phases[1]: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
}

image_datasets = {
    phase: ImageFolder(
        os.path.join(folder, phase),
        transform=image_transforms[phase],
    ) for phase in phases
}
image_dataloaders = {
    phase: DataLoader(
        dataset=image_datasets[phase],
        batch_size=batch_size,
        shuffle=(phase == 'train')
    ) for phase in phases
}

dataset_sizes = {
    phase: len(image_datasets[phase])
    for phase in phases
}
classes = image_datasets['train'].classes


def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


print()

print(f"classes = {classes}")
print(f"type(classes) = {type(classes)}")

print()

train_images, train_labels = next(iter(image_dataloaders['train']))
print(f"len(image_datasets['train']) = {len(image_datasets['train'])}")
print(f"len(image_dataloaders['train'] = {len(image_dataloaders['train'])}")
print(f"train_images.shape = {train_images.shape}")
print(f"train_labels.shape = {train_labels.shape}")
print(f"train_labels = {train_labels}")
# inp = make_grid(train_images)
# imshow(inp, [classes[label] for label in train_labels])

print()

val_images, val_labels = next(iter(image_dataloaders['val']))
print(f"len(image_datasets['val']) = {len(image_datasets['val'])}")
print(f"len(image_dataloaders['val']) = {len(image_dataloaders['val'])}")
print(f"val_images.shape = {val_images.shape}")
print(f"val_labels.shape = {val_labels.shape}")
print(f"val_labels = {val_labels}")
# inp = make_grid(val_images)
# imshow(inp, [classes[label] for label in val_labels])

print()


def train(model, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    best_wts = deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        for phase in phases:
            cum_loss = 0.0
            cum_acc = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for images, labels in image_dataloaders[phase]:
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
                    cum_loss += loss.item() * images.shape[0]
                    cum_acc += torch.sum(predictions == labels).item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = cum_loss / dataset_sizes[phase]
            epoch_acc = cum_acc / dataset_sizes[phase]

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = deepcopy(model.state_dict())

            print(f"epoch {epoch+1}/{num_epochs}, loss {epoch_loss}, accuracy {epoch_acc}")  # noqa: E501

    model.load_state_dict(best_wts)
    return model
