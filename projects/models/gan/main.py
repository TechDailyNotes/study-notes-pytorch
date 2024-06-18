import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer_fake = SummaryWriter('runs/gan_mnist/fake')
writer_real = SummaryWriter('runs/gan_mnist_real')

batch_size = 4

z_features = 64
img_features = 28 * 28 * 1  # 784
dis_negslope = 0.1
gen_negslope = 0.1
lr = 3e-4

num_epochs = 2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = datasets.MNIST('data', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True)

print('Data Loading Completed!')


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


dis = Discriminator(img_features, 128).to(device)
gen = Generator(z_features, 256, img_features).to(device)
criterion = nn.BCELoss()
optim_dis = optim.Adam(dis.parameters(), lr)
optim_gen = optim.Adam(gen.parameters(), lr)

num_batches = len(dataloader)
test_img = torch.rand((batch_size, z_features)).to(device)
test_step = 0

print('Model Training Started!')

for epoch in range(num_epochs):
    for batch, (real_img, _) in enumerate(dataloader):
        real_img = real_img.reshape(real_img.shape[0], -1).to(device)
        noise_img = torch.randn((batch_size, z_features)).to(device)

        dis_real = dis(real_img).reshape(-1)
        loss_real = criterion(dis_real, torch.ones_like(dis_real))

        fake_img = gen(noise_img)
        dis_fake = dis(fake_img).reshape(-1)
        loss_fake = criterion(dis_fake, torch.zeros_like(dis_fake))

        loss_dis = (loss_real + loss_fake) / 2
        optim_dis.zero_grad()
        loss_dis.backward(retain_graph=True)
        optim_dis.step()

        output = dis(fake_img).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch + 1 == num_batches:
            print(
                f"epoch {epoch + 1}/{num_epochs}, "
                f"discriminator loss {loss_dis.item():.4f}, "
                f"generator loss {loss_gen.item():.4f}"
            )

            with torch.no_grad():
                fake_img = gen(test_img).reshape(-1, 1, 28, 28)
                real_img = real_img.reshape(-1, 1, 28, 28)

                fake_img_grid = make_grid(fake_img, normalize=True)
                real_img_grid = make_grid(real_img, normalize=True)

                writer_fake.add_image(
                    'MNIST Fake Images', fake_img_grid, test_step
                )
                writer_real.add_image(
                    'MNIST Real Images', real_img_grid, test_step
                )
                test_step += 1

print('Model Training Completed!')
