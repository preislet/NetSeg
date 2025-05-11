import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.down1 = ConvBlock(in_channels, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up3 = ConvBlock(512 + 256, 256)
        self.up2 = ConvBlock(256 + 128, 128)
        self.up1 = ConvBlock(128 + 64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(self.pool(a))
        c = self.down3(self.pool(b))
        d = self.down4(self.pool(c))

        d_up = F.interpolate(d, scale_factor=2, mode='nearest')
        e = self.up3(torch.cat([d_up, c], dim=1))

        e_up = F.interpolate(e, scale_factor=2, mode='nearest')
        f = self.up2(torch.cat([e_up, b], dim=1))

        f_up = F.interpolate(f, scale_factor=2, mode='nearest')
        y = self.up1(torch.cat([f_up, a], dim=1))

        return self.final(y)