import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Residual(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, downsample=None):
        super(Residual, self).__init__()

        md_chans = out_chans // 4
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_chans, md_chans, 1)
        self.bn1 = nn.BatchNorm2d(md_chans)
        self.conv2 = nn.Conv2d(md_chans, md_chans, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(md_chans)
        self.conv3 = nn.Conv2d(md_chans, out_chans, 1)
        self.bn3 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, img_chans=3, num_classes=1000):
        super(ResNet, self).__init__()

        # Layer Group 1: Convolution
        self.conv = nn.Conv2d(img_chans, 64, 7, 2, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Layer Group 2: Residual Connection
        layers = []
        in_chans, out_chans, stride = 64, 256, 1

        for num_layer in num_layers:
            # Step 2.1: Append the downsample layer
            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, stride),
                nn.BatchNorm2d(out_chans),
            )
            layers.append(Residual(in_chans, out_chans, stride, downsample))

            # Step 2.2: Append the rest layers
            in_chans = out_chans
            for _ in range(num_layer - 1):
                layers.append(Residual(in_chans, out_chans))

            # Step 2.3: Update Residual's parameters
            in_chans, out_chans, stride = out_chans, out_chans * 2, 2

        self.residual = nn.Sequential(*layers)

        # Layer Group 3: Feed Forward
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.maxpool(out)
        out = self.residual(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


x = torch.randn((4, 3, 224, 224), dtype=torch.float32).to(device)

model50 = ResNet([3, 4, 6, 3])
out50 = model50(x)
print(f"out50.shape = {out50.shape}")

model101 = ResNet([3, 4, 23, 3])
out101 = model101(x)
print(f"out101.shape = {out101.shape}")

model152 = ResNet([3, 8, 36, 3])
out152 = model152(x)
print(f"out152.shape = {out152.shape}")
