import math
import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

KERNELS = [
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
]

STRIDES = [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]

DYNUNET_CONFIG = {
    "spatial_dims": 3,
    "in_channels": 3,
    "out_channels": 1,
    "kernel_size": KERNELS,
    "strides": STRIDES,
    "upsample_kernel_size": STRIDES[1:],
    "filters": (32, 64, 128, 256, 320),
    "norm_name": "instance",
    "act_name": ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    "deep_supervision": True,
    "deep_supr_num": 2,
    "res_block": True,
    "dropout": None,
}

TUMOUR_PREVALENCE = 0.0005
OUTPUT_BIAS_INIT = math.log(TUMOUR_PREVALENCE / (1.0 - TUMOUR_PREVALENCE))


def get_dynunet(device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynUNet(**DYNUNET_CONFIG).to(device)

    # ===================== FIXED BIAS LOGIC ===================== #

    target_bias_name  = None
    target_bias_param = None

    # PRIORITY: main output layer
    for name, param in model.named_parameters():
        if "output_block" in name and "bias" in name and param.dim() == 1:
            target_bias_name  = name
            target_bias_param = param
            break

    # fallback
    if target_bias_param is None:
        for name, param in model.named_parameters():
            if "bias" in name and param.dim() == 1:
                target_bias_name  = name
                target_bias_param = param

    if target_bias_param is not None:
        with torch.no_grad():
            target_bias_param.fill_(OUTPUT_BIAS_INIT)

        init_sigmoid = torch.sigmoid(torch.tensor(OUTPUT_BIAS_INIT)).item()

        print(f"  Output bias layer : {target_bias_name}")
        print(f"  Output bias init  : {OUTPUT_BIAS_INIT:.4f}"
              f"  (sigmoid = {init_sigmoid:.5f}  ≈ tumour prevalence)")
    else:
        print("  WARNING: could not find output bias — collapse fix NOT applied")

    # ============================================================ #

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model : DynUNet 3D (deep supervision)")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable        : {train_params:,}")
    print(f"  Device           : {device}")

    return model