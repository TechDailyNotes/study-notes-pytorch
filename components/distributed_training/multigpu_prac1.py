import argparse
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datautils import MyTrainDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def get_train_objs(batch_size, learning_rate):
    dataset = MyTrainDataset(4096)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
    )
    model = nn.Linear(20, 1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    return dataloader, model, optimizer


class Trainer:
    def __init__(
        self,
        rank,
        world_size,
        dataloader,
        model,
        optimizer,
        total_epochs,
        save_every,
    ):
        self.rank = rank
        self.world_size = world_size
        self.dataloader = dataloader
        self.model = DDP(model.to(rank), device_ids=[rank])
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
        self.dataloader.sampler.set_epoch(epoch_id)
        for batch_id, (source, target) in enumerate(self.dataloader):
            source = source.to(self.rank)
            target = target.to(self.rank)
            self._run_batch(source, target)

            print(
                f"rank {self.rank}/{torch.cuda.device_count()}, "
                f"epoch {epoch_id}/{self.total_epochs}, "
                f"batch {batch_id}/{len(self.dataloader)}, "
                f"batch size {len(next(iter(self.dataloader))[0])}"
            )

    def _save_checkpoint(self, epoch_id):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/checkpoint.pt")
        print(f"epoch {epoch_id}/{self.total_epochs}, saving checkpoints to checkpoints/checkpoint.pt")

    def train(self):
        for epoch_id in range(self.total_epochs):
            self._run_epoch(epoch_id)

            if self.rank == 0 and epoch_id % self.save_every == 0:
                self._save_checkpoint(epoch_id)


def main(rank, world_size, batch_size, learning_rate, total_epochs, save_every):
    ddp_setup(rank, world_size)
    dataloader, model, optimizer = get_train_objs(batch_size, learning_rate)
    trainer = Trainer(rank, world_size, dataloader, model, optimizer, total_epochs, save_every)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help="Learning rate of the optimizer (default: 1e-3)")
    parser.add_argument('-e', '--total_epochs', default=2, type=int, help='Total epochs to train the model')
    parser.add_argument('-s', '--save_every', default=2, type=int, help='How often to save a snapshot')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.batch_size, args.learning_rate, args.total_epochs, args.save_every), nprocs=world_size)
