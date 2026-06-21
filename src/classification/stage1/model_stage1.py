"""
model_stage1.py
================
Stage 1 Tumor Detection Model: 3D ResNet18 + CBAM Attention

Architecture:
  Input (B, 3, D, H, W)  — any spatial size, size-agnostic
    → Stem: Conv3d 7×7×7 stride 2 → BN → ReLU → MaxPool
    → Layer1: 2× BasicBlock3D (64 ch,  no CBAM)
    → Layer2: 2× BasicBlock3D (128 ch, + CBAM)
    → Layer3: 2× BasicBlock3D (256 ch, + CBAM)
    → Layer4: 2× BasicBlock3D (512 ch, + CBAM)
    → AdaptiveAvgPool3d(1,1,1) → flatten
    → Dropout(0.4) → FC 512→256 → ReLU → Dropout(0.3) → FC 256→1
  Output: raw logit scalar (no sigmoid — use BCEWithLogitsLoss)

Parameters: ~33.4M
VRAM estimate:
  RTX 3050 6GB:  batch=2, input 141×213×213 → ~2.0 GB  ✓
  RTX 2000 Ada:  batch=4, input 141×213×213 → ~3.5 GB  ✓
"""

import torch
import torch.nn as nn


# ── Channel Attention (Squeeze-and-Excitation style) ──────────────

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        return x * scale


# ── Spatial Attention ─────────────────────────────────────────────

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx    = x.max(dim=1, keepdim=True).values
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


# ── CBAM: Channel + Spatial ───────────────────────────────────────

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention3D(channels, reduction)
        self.spatial = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ── 3D BasicBlock ─────────────────────────────────────────────────

class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.cbam  = CBAM3D(out_ch) if use_cbam else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam is not None:
            out = self.cbam(out)
        return self.relu(out + self.shortcut(x))


# ── 3D ResNet18 + CBAM ────────────────────────────────────────────

class ResNet3D_Stage1(nn.Module):
    """
    3D ResNet18 with CBAM on layers 2–4.
    Size-agnostic: works on any input volume size ≥ 16×16×16.
    """

    def __init__(self, in_channels=3, dropout=0.4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1, cbam=False)
        self.layer2 = self._make_layer(64,  128, 2, stride=2, cbam=True)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, cbam=True)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, cbam=True)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n, stride, cbam):
        layers = [BasicBlock3D(in_ch, out_ch, stride=stride, use_cbam=cbam)]
        for _ in range(1, n):
            layers.append(BasicBlock3D(out_ch, out_ch, stride=1,
                                        use_cbam=cbam))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).flatten(1)
        return self.head(x)   # (B, 1) raw logit

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(in_channels=3, dropout=0.4):
    model = ResNet3D_Stage1(in_channels=in_channels, dropout=dropout)
    n = model.count_parameters()
    print(f"  Stage1 ResNet3D+CBAM: {n/1e6:.1f}M parameters")
    return model


if __name__ == "__main__":
    model = build_model()
    for shape in [(1,3,64,128,128), (1,3,141,213,213), (1,3,50,181,179)]:
        x   = torch.randn(*shape)
        out = model(x)
        print(f"  Input {str(shape):30s} → output {tuple(out.shape)}")
