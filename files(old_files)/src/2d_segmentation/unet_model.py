import torch
import torch.nn as nn


# -----------------------------
# Attention Block
# -----------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# -----------------------------
# Double Conv Block
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):  # FIX: 0.2->0.1
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# Attention U-Net
# -----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.down1 = DoubleConv(1, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bridge
        self.bridge = DoubleConv(512, 1024)

        # Decoder + Attention
        self.up4   = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4  = AttentionBlock(512, 512, 256)
        self.conv4 = DoubleConv(1024, 512)

        self.up3   = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3  = AttentionBlock(256, 256, 128)
        self.conv3 = DoubleConv(512, 256)

        self.up2   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2  = AttentionBlock(128, 128, 64)
        self.conv2 = DoubleConv(256, 128)

        self.up1   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1  = AttentionBlock(64, 64, 32)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

        # Weight initialization — helps converge faster
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Encoder
        d1 = self.down1(x);  p1 = self.pool(d1)
        d2 = self.down2(p1); p2 = self.pool(d2)
        d3 = self.down3(p2); p3 = self.pool(d3)
        d4 = self.down4(p3); p4 = self.pool(d4)

        # Bridge
        bridge = self.bridge(p4)

        # Decoder with Attention
        up4    = self.up4(bridge)
        d4     = self.att4(up4, d4)
        c4     = self.conv4(torch.cat([up4, d4], dim=1))

        up3    = self.up3(c4)
        d3     = self.att3(up3, d3)
        c3     = self.conv3(torch.cat([up3, d3], dim=1))

        up2    = self.up2(c3)
        d2     = self.att2(up2, d2)
        c2     = self.conv2(torch.cat([up2, d2], dim=1))

        up1    = self.up1(c2)
        d1     = self.att1(up1, d1)
        c1     = self.conv1(torch.cat([up1, d1], dim=1))

        return self.final(c1)