import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datautils import MyTrainDataset


class Trainer:
    def __init__(self, gpu_id, data, model, optimizer, save_every):
        self.gpu_id = gpu_id
        self.data = data
        self.model = model.to(gpu_id)
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch_id):
        print(f"gpu_id = {self.gpu_id}, epoch = {epoch_id}, batch_size = {len(next(iter(self.data))[0])}, batch_num = {len(self.data)}")
        for source, target in self.data:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(source, target)

    def train(self, total_epochs):
        for epoch_id in range(total_epochs):
            self._run_epoch(epoch_id)


def load_train_objs():
    dataset = MyTrainDataset(2048)
    model = nn.Linear(20, 1)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return dataset, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )


def main(device, batch_size, total_epochs, save_every):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(device, train_data, model, optimizer, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-t', '--total_epochs', default=2, type=int, help='Total epochs to train the model')
    parser.add_argument('-s', '--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.batch_size, args.total_epochs, args.save_every)
