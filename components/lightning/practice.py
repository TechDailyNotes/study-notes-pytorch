import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from pytorch_lightning import Trainer


input_size = 784
hidden_size = 500
num_classes = 10
batch_size = 100
lr = 0.01
num_epochs = 2


class LitFFN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitFFN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.validation_step_outputs = []

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=lr)

    def train_dataloader(self):
        dataset = MNIST(
            './data', train=True, transform=ToTensor(), download=True,
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            num_workers=11, persistent_workers=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = MNIST(
            './data', train=False, transform=ToTensor(), download=False,
        )
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False,
            num_workers=11, persistent_workers=True,
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, input_size)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, input_size)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        tensorboard_logs = {'val_loss': loss}
        self.validation_step_outputs.append(loss)
        return {'val_loss': loss, 'log': tensorboard_logs}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


if __name__ == '__main__':
    trainer = Trainer(fast_dev_run=False, max_epochs=num_epochs)
    model = LitFFN(input_size, hidden_size, num_classes)
    trainer.fit(model)
