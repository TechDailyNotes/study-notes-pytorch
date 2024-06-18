from model import Discriminator, Generator, initialize_weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer_fake = SummaryWriter("runs/fake")
writer_real = SummaryWriter("runs/real")

batch_size = 4

z_chans, img_chans, img_height, img_width = 100, 1, 64, 64
dis_hidden_chans, gen_hidden_chans = 64, 64
lr = 2e-4

num_epochs = 2

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(img_chans)],
        [0.5 for _ in range(img_chans)],
    ),
])
dataset = MNIST('data', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
num_batches = len(dataloader)

print("Data Loading Succeeded!")

dis = Discriminator(img_chans, dis_hidden_chans).to(device)
gen = Generator(z_chans, gen_hidden_chans, img_chans).to(device)
initialize_weights(dis)
initialize_weights(gen)

criterion = nn.BCELoss()
optim_dis = optim.Adam(dis.parameters(), lr, (0.5, 0.999))
optim_gen = optim.Adam(gen.parameters(), lr, (0.5, 0.999))

print("Model Training Launched!")

test_img_noise = torch.randn((batch_size, z_chans, 1, 1)).to(device)
global_step = 0

for epoch in range(num_epochs):
    for batch, (img_real, _) in enumerate(dataloader):
        # Step 0: Training setup.
        dis.train()
        gen.train()

        # Step 1: Train the discriminator.
        img_real = img_real.to(device)
        dis_real = dis(img_real).reshape(-1)
        loss_real = criterion(dis_real, torch.ones_like(dis_real))

        img_noise = torch.randn((batch_size, z_chans, 1, 1)).to(device)
        img_fake = gen(img_noise)
        dis_fake = dis(img_fake).reshape(-1)
        loss_fake = criterion(dis_fake, torch.zeros_like(dis_fake))

        loss_dis = (loss_real + loss_fake) / 2
        optim_dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        optim_dis.step()

        # Step 2: Train the generator.
        dis_fake = dis(img_fake).reshape(-1)
        loss_gen = criterion(dis_fake, torch.ones_like(dis_fake))
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # Step 3: Result Inference
        if (batch + 1) % 100 == 0:
            print(
                f"epoch {epoch + 1}/{num_epochs}, "
                f"batch {batch + 1}/{num_batches}, "
                f"generator loss {loss_gen.item():.4f}, "
                f"discriminator loss {loss_dis.item():.4f}"
            )

            with torch.no_grad():
                test_img_fake = gen(test_img_noise)
                test_img_fake_grid = make_grid(test_img_fake, normalize=True)
                img_real_grid = make_grid(img_real, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", test_img_fake_grid, global_step,
                )
                writer_real.add_image(
                    "MNIST Real Images", img_real_grid, global_step
                )
                global_step += 1

print("Model Training Succeeded!")
