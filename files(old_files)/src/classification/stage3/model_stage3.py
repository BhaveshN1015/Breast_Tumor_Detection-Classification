"""
model_stage3.py
===============
3D EfficientNet-B0 for Stage 3 — Benign vs Malignant classification.

FIX v2 — Output bias corrected:
  Previous: OUTPUT_BIAS = log(136/323) = -0.865
            sigmoid(-0.865) = 29.6% → all outputs below threshold 0.5
            → AUC=0.5 for first 10 epochs (model couldn't discriminate)
  Now: OUTPUT_BIAS = 0.0 (neutral 50/50 start)
       Combined with corrected pos_weight=0.403 the loss converges cleanly.

Dataset (training split): 221 malignant (71%)  /  89 benign (29%)
Malignant IS the majority class — the loss pos_weight must DOWNWEIGHT it:
  pos_weight = N_benign / N_malignant = 89/221 = 0.403
  (was 2.375 = 323/136 total dataset ratio — wrong direction, caused collapse)
"""

import math
import torch
import torch.nn as nn

OUTPUT_BIAS = 0.0   # neutral: sigmoid(0) = 50% — loss steers from here


def build_model(dropout=0.4, device=None):
    """Factory matching Stage 1 build_model() interface."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return get_model_stage3(device, dropout=dropout)


def get_model_stage3(device, dropout=0.4):
    try:
        from monai.networks.nets import EfficientNetBN
        model = EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=3,
            in_channels=3,
            num_classes=1,
        )
        in_features = model._fc.in_features
        model._fc   = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        model._fc[1].bias.data.fill_(OUTPUT_BIAS)
        n = _count_params(model)
        print(f"  Stage3 Model : 3D EfficientNet-B0 (MONAI) — {n:.1f}M params")
        print(f"  Output bias  : {OUTPUT_BIAS:.4f}  "
              f"(neutral 50/50 prior — loss corrects via pos_weight=0.403)")
        return model.to(device)

    except Exception as e:
        print(f"  MONAI EfficientNetBN unavailable ({e}). Using fallback 3D CNN.")
        return _Fallback3DCNN(dropout=dropout).to(device)


class _ConvBnSiLU(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class _Fallback3DCNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBnSiLU(3, 32, stride=2), _ConvBnSiLU(32, 64, stride=2),
            _ConvBnSiLU(64, 128, stride=2), _ConvBnSiLU(128, 256, stride=2),
            _ConvBnSiLU(256, 320, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout), nn.Linear(320, 1))
        self.head[-1].bias.data.fill_(OUTPUT_BIAS)
        print(f"  Stage3 Model : Fallback 3D CNN — {_count_params(self):.1f}M params")

    def forward(self, x): return self.head(self.pool(self.encoder(x)))
    def count_parameters(self): return _count_params(self)


def _count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6