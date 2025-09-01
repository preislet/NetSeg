import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Building blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Gating on skip features x by decoder features g.
    Projects both to 'inter_ch', sums, ReLU, 1x1 -> sigmoid to get mask.
    """
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        a  = F.relu(g1 + x1, inplace=True)
        m  = self.psi(a)          # (B,1,H,W)
        return x * m              # gate skip


# -------------------------
# Attention U-Net (size-safe)
# -------------------------
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder + Attention gates
        # (we upsample with interpolate -> exact size match to skip)
        self.att4 = AttentionGate(g_ch=512, x_ch=512, inter_ch=256)
        self.dec4 = ConvBlock(1024, 512)

        self.att3 = AttentionGate(g_ch=256, x_ch=256, inter_ch=128)
        self.dec3 = ConvBlock(512, 256)

        self.att2 = AttentionGate(g_ch=128, x_ch=128, inter_ch=64)
        self.dec2 = ConvBlock(256, 128)

        self.att1 = AttentionGate(g_ch=64, x_ch=64, inter_ch=32)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def upsample_to(x, ref):
        """Upsample x to ref spatial size (H,W) exactly."""
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                # 64
        e2 = self.enc2(self.pool(e1))    # 128
        e3 = self.enc3(self.pool(e2))    # 256
        e4 = self.enc4(self.pool(e3))    # 512

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))  # 1024

        # Decoder stage 4 (to e4 size)
        d4 = self.upsample_to(b, e4)         # 1024 -> size of e4
        g4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, g4], dim=1))  # 1024 -> 512

        # Decoder stage 3 (to e3 size)
        d3 = self.upsample_to(d4, e3)        # 512 -> size of e3
        g3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, g3], dim=1))  # 512 -> 256

        # Decoder stage 2 (to e2 size)
        d2 = self.upsample_to(d3, e2)        # 256 -> size of e2
        g2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, g2], dim=1))  # 256 -> 128

        # Decoder stage 1 (to e1 size)
        d1 = self.upsample_to(d2, e1)        # 128 -> size of e1
        g1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, g1], dim=1))  # 128 -> 64

        return self.final(d1)
