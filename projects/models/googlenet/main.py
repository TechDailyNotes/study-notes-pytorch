import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.conv1 = Conv(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = Conv(64, 192, kernel_size=3, stride=1, padding=1)
        self.incp3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
        )
        self.incp4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
        )
        self.incp5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.maxpool(self.conv1(x))
        out = self.maxpool(self.conv2(out))
        out = self.maxpool(self.incp3(out))
        out = self.maxpool(self.incp4(out))
        out = self.avgpool(self.incp5(out))
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class Inception(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5,
        out_poolproj,
    ):
        super(Inception, self).__init__()
        self.branch1 = Conv(
            in_channels, out_1x1, kernel_size=1, stride=1, padding=0,
        )
        self.branch2 = nn.Sequential(
            Conv(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
            Conv(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
            Conv(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, out_poolproj, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        ], dim=1)
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


x = torch.randn((4, 3, 224, 224), dtype=torch.float32).to(device)
model = GoogLeNet().to(device)
out = model(x)
print(f"out.shape = {out.shape}")
