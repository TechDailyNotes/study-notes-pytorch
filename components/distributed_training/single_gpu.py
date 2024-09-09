import torch.functional as F
import torch.nn as nn
import torch.optimizer as optim
from torch.utils.data import DataLoader
from datautils import MyTrainDataset


def load_train_objs():
    dataset = MyTrainDataset(2048)
    model = nn.Linear(20, 1)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return dataset, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        
    )


def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-t', '--total_epochs', default=2, type=int, help='Total epochs to train the model')
    parser.add_argument('-s', '--save_every', default=2, type=int, help='How often to save a snapshot')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
