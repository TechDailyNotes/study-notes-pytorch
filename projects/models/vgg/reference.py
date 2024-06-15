import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_arch = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M',
        ]

        self.convs = self.get_convs()
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fcs(out)
        return out

    def get_convs(self):
        convs = []
        in_channels = self.in_channels

        for x in self.conv_arch:
            if type(x) is int:
                convs.extend([
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=x,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ])
                in_channels = x
            else:
                convs.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        return nn.Sequential(*convs)


x = torch.randn((32, 3, 224, 224), dtype=torch.float32).to(device)
model = VGG().to(device)
out = model(x)
print(f"out.shape = {out.shape}")
