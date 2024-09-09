import argparse
import torch.nn.functional as F
from datautils import MyTrainDataset
from torch.nn import Linear
from torch.optim import SGD
from torch.utils.data import DataLoader


def get_train_objs(batch_size, learning_rate):
    dataset = MyTrainDataset(2048)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    model = Linear(20, 1)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    return dataloader, model, optimizer


class Trainer:
    def __init__(self, gpu_id, dataloader, model, optimizer, total_epochs, save_every):
        self.gpu_id = gpu_id
        self.dataloader = dataloader
        self.model = model.to(gpu_id)
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.save_every = save_every

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch_id):
        for batch_id, (source, target) in enumerate(self.dataloader):
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target)

            print(f"device {self.gpu_id}, epoch {epoch_id}/{self.total_epochs}, batch {batch_id}/{len(self.dataloader)}")

    def train(self):
        for epoch_id in range(self.total_epochs):
            self._run_epoch(epoch_id)


def main(gpu_id, batch_size, learning_rate, total_epochs, save_every):
    dataloader, model, optimizer = get_train_objs(batch_size, learning_rate)
    trainer = Trainer(gpu_id, dataloader, model, optimizer, total_epochs, save_every)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='Learning rate of the optimizer (default" 1e-3)')
    parser.add_argument('-t', '--total_epochs', default=2, type=int, help='Total epochs to train the model (default: 2)')
    parser.add_argument('-s', '--save_every', default=2, type=int, help='How often to save a snapshot (default: 2)')
    args = parser.parse_args()

    gpu_id = 0
    main(gpu_id, args.batch_size, args.learning_rate, args.total_epochs, args.save_every)
