"""
model_3d.py — upgraded for combined dataset training
=====================================================

Key upgrades from previous version:
  - OUTPUT BIAS CORRECTION unchanged — sigmoid(-7.60) = 0.0005.
  - NEW: keep_largest_component() helper — call from predict_3d.py to
    remove satellite false-positive blobs (fixes patient 204 class).
  - NEW: post_process_prediction() convenience wrapper — applies
    morphological closing + min-size filter + keep largest component.
  - Architecture unchanged: channels (32,64,128,256,320), 12.8M params.
    This is correct for RTX 2000 Ada 16GB.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from monai.networks.nets import UNet as MonaiUNet


# ------------------------------------------------------------------ #
#  ARCHITECTURE PARAMETERS                                             #
# ------------------------------------------------------------------ #

MODEL_CONFIG = {
    "spatial_dims" : 3,
    "in_channels"  : 3,           # P1 + P2 + P3
    "out_channels" : 1,           # binary tumour mask
    "channels"     : (32, 64, 128, 256, 320),   # RTX 2000 Ada 16GB
    "strides"      : (2, 2, 2, 2),
    "num_res_units": 2,
    "norm"         : "instance",
    "act"          : ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    "dropout"      : 0.1,
}

# ------------------------------------------------------------------ #
#  OUTPUT BIAS                                                         #
# ------------------------------------------------------------------ #

TUMOR_PREVALENCE = 0.0005
OUTPUT_BIAS_INIT = math.log(TUMOR_PREVALENCE / (1.0 - TUMOR_PREVALENCE))  # ≈ -7.60


def get_model(device=None):
    """Instantiate MONAI ResNet UNet 3D with output bias correction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MonaiUNet(**MODEL_CONFIG).to(device)

    # Find final conv bias and initialise to OUTPUT_BIAS_INIT
    last_bias_name  = None
    last_bias_param = None
    for name, param in model.named_parameters():
        if 'bias' in name and param.dim() == 1:
            last_bias_name  = name
            last_bias_param = param

    if last_bias_param is not None:
        with torch.no_grad():
            last_bias_param.fill_(OUTPUT_BIAS_INIT)
        init_sig = torch.sigmoid(torch.tensor(OUTPUT_BIAS_INIT)).item()
        print(f"  Output bias layer : {last_bias_name}")
        print(f"  Output bias init  : {OUTPUT_BIAS_INIT:.4f}"
              f"  (sigmoid = {init_sig:.5f}  ≈ tumor prevalence)")
    else:
        print("  WARNING: could not find final bias — collapse fix NOT applied")

    total  = sum(p.numel() for p in model.parameters())
    trainp = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model : MONAI ResNet UNet 3D")
    print(f"  spatial_dims  : {MODEL_CONFIG['spatial_dims']}")
    print(f"  in_channels   : {MODEL_CONFIG['in_channels']}  (P1, P2, P3)")
    print(f"  channels      : {MODEL_CONFIG['channels']}")
    print(f"  strides       : {MODEL_CONFIG['strides']}")
    print(f"  num_res_units : {MODEL_CONFIG['num_res_units']}")
    print(f"  norm          : {MODEL_CONFIG['norm']}")
    print(f"  dropout       : {MODEL_CONFIG['dropout']}")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainp:,}")
    print(f"  Device        : {device}")

    return model


# ------------------------------------------------------------------ #
#  POST-PROCESSING HELPERS                                             #
# ------------------------------------------------------------------ #

def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keeps only the single largest connected component.
    Removes all satellite blobs — critical for patient 204 class where
    the model over-predicts 13k voxels around a 768-voxel GT tumor.

    Args:
        binary_mask: np.ndarray of shape (D, H, W), dtype uint8 or bool
    Returns:
        np.ndarray same shape, dtype uint8, only largest component kept
    """
    labeled, n_features = ndimage.label(binary_mask)
    if n_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    sizes = ndimage.sum(binary_mask, labeled, range(1, n_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return (labeled == largest_label).astype(np.uint8)


def post_process_prediction(binary_mask: np.ndarray,
                             min_size_voxels: int = 50,
                             use_largest_only: bool = True) -> np.ndarray:
    """
    Full post-processing pipeline for a binary prediction mask.

    Steps:
      1. Morphological closing (fills small holes, smooths boundary)
      2. Remove connected components smaller than min_size_voxels
      3. Optionally keep only the single largest component

    Args:
        binary_mask     : (D, H, W) uint8 or bool
        min_size_voxels : minimum blob size to keep (default 50)
                          Set to 50 — not 200 — so small real tumors are kept.
                          keep_largest_component handles the over-prediction
                          issue better than a high min_size threshold.
        use_largest_only: if True, keep only the largest component after
                          size filtering. Set True for inference — breast
                          tumors are almost always a single connected mass.
    Returns:
        (D, H, W) uint8 cleaned mask
    """
    if binary_mask.sum() == 0:
        return binary_mask.astype(np.uint8)

    # Step 1: morphological closing
    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(binary_mask, structure=struct, iterations=2)

    # Step 2: remove small blobs
    labeled, n = ndimage.label(closed)
    cleaned = np.zeros_like(closed, dtype=np.uint8)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_size_voxels:
            cleaned[labeled == i] = 1

    if cleaned.sum() == 0:
        return cleaned

    # Step 3: keep largest component only
    if use_largest_only:
        cleaned = keep_largest_component(cleaned)

    return cleaned
