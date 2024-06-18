import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fake_writer = SummaryWriter('runs/practice/fake')
real_writer = SummaryWriter('runs/practice/real')

img_mean = (0.5,)
img_std = (0.5,)
batch_size = 4

z_features = 64
img_features = 1 * 28 * 28
dis_negslope = 0.1
gen_negslope = 0.1
lr = 3e-4

num_epochs = 2

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(img_mean, img_std),
])
dataset = MNIST('data', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)
num_batches = len(dataloader)


class Generator(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(gen_negslope),
            nn.Linear(hidden_features, out_features),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.generator(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(dis_negslope),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.discriminator(x)
        return out


gen = Generator(z_features, 256, img_features).to(device)
dis = Discriminator(img_features, 128).to(device)
criterion = nn.BCELoss()
optim_gen = optim.Adam(gen.parameters(), lr=lr)
optim_dis = optim.Adam(dis.parameters(), lr=lr)

img_test = torch.randn((batch_size, z_features)).to(device)
global_step = 0

for epoch in range(num_epochs):
    for batch, (img_real, _) in enumerate(dataloader):
        # Step 1: Train discriminator.
        img_real = img_real.reshape(img_real.shape[0], -1).to(device)
        dis_real = dis(img_real).reshape(-1)

        img_noise = torch.randn((batch_size, z_features)).to(device)
        img_fake = gen(img_noise)
        dis_fake = dis(img_fake).reshape(-1)

        loss_real = criterion(dis_real, torch.ones_like(dis_real))
        loss_fake = criterion(dis_fake, torch.zeros_like(dis_fake))
        loss_dis = (loss_real + loss_fake) / 2

        optim_dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        optim_dis.step()

        # Step 2: Train generator.
        output = dis(img_fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # Step 3: Log results.
        if batch + 1 == num_batches:
            print(
                f"epoch {epoch + 1}/{num_epochs}, "
                f"generator loss {loss_gen.item():.4f}, "
                f"discriminator loss {loss_dis.item():.4f}"
            )

            img_real = img_real.reshape(-1, 1, 28, 28)
            img_fake = gen(img_test).reshape(-1, 1, 28, 28)

            img_real_grid = make_grid(img_real, normalize=True)
            img_fake_grid = make_grid(img_fake, normalize=True)

            real_writer.add_image(
                'MNIST Real Images', img_real_grid, global_step,
            )
            fake_writer.add_image(
                'MNIST Fake Images', img_fake_grid, global_step,
            )
            global_step += 1
