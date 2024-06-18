import torch
import torch.nn as nn


dis_negslope = 0.2


class DisConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(DisConv, self).__init__()
        self.disconv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(dis_negslope),
        )

    def forward(self, x):
        return self.disconv(x)


class Discriminator(nn.Module):
    def __init__(self, in_chans, hidden_chans):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, 4, 2, 1),  # 32 * 32
            nn.LeakyReLU(dis_negslope),
            DisConv(hidden_chans, hidden_chans * 2, 4, 2, 1),  # 16 * 16
            DisConv(hidden_chans * 2, hidden_chans * 4, 4, 2, 1),  # 8 * 8
            DisConv(hidden_chans * 4, hidden_chans * 8, 4, 2, 1),  # 4 * 4
            nn.Conv2d(hidden_chans * 8, 1, 4, 2, 0),  # 1 * 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)


class GenConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(GenConv, self).__init__()
        self.genconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.genconv(x)


class Generator(nn.Module):
    def __init__(self, in_chans, hidden_chans, out_chans):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            GenConv(in_chans, hidden_chans * 16, 4, 1, 0),  # 4*4
            GenConv(hidden_chans * 16, hidden_chans * 8, 4, 2, 1),  # 8*8
            GenConv(hidden_chans * 8, hidden_chans * 4, 4, 2, 1),  # 16*16
            GenConv(hidden_chans * 4, hidden_chans * 2, 4, 2, 1),  # 32*32
            nn.ConvTranspose2d(hidden_chans * 2, out_chans, 4, 2, 1),  # 64*64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    print("Model Test Launched!")

    # Step 0: Basic parameters setup.
    batch_size, z_chans, img_chans, img_height, img_width = 4, 100, 3, 64, 64
    dis_hidden_chans, gen_hidden_chans = 8, 8

    # Step 1: Test the discriminator.
    test_img = torch.randn((batch_size, img_chans, img_height, img_width))
    dis = Discriminator(img_chans, dis_hidden_chans)
    initialize_weights(dis)
    assert dis(test_img).shape == (batch_size, 1, 1, 1)

    # Step 2: Test the generator.
    test_z = torch.randn((batch_size, z_chans, 1, 1))
    gen = Generator(z_chans, gen_hidden_chans, img_chans)
    initialize_weights(gen)
    assert gen(test_z).shape == (batch_size, img_chans, img_height, img_width)

    print("Model Test Succeeded!")


if __name__ == '__main__':
    test()
