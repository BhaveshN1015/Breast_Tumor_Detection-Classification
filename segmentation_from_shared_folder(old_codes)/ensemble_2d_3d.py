"""
ensemble_2d_3d.py
=================
Combines your existing 2D ResNet34 U-Net with the new 3D MONAI ResNet UNet
to produce an ensemble prediction with higher Dice than either model alone.

How it works:
  1. Load 3D probability volumes saved by predict_3d.py (per patient .nii.gz)
  2. For each test patient, re-run 2D inference slice-by-slice and stack
     into a matching 3D probability volume
  3. Weighted average the two probability volumes
  4. Threshold → final ensemble binary mask
  5. Evaluate and report improvement over both individual models

Weight sweep is performed on validation patients to find the optimal
weighting — same data-driven approach as your 2D threshold sweep.

Usage:
    python src/segmentation_3d/ensemble_2d_3d.py

Requirements:
  - predict_3d.py must have been run first (3D predictions saved)
  - Your 2D model must be accessible at MODEL_2D_PATH
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
import cv2
from torch.cuda.amp import autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp
from monai.inferers import sliding_window_inference

sys.path.append(os.path.dirname(__file__))
sys.path.append("src/segmentation")   # for your 2D pipeline utilities

from data_3d  import get_val_transforms, _expand_paths, PATCH_SIZE
from model_3d import get_model as get_3d_model
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiLoader

# -----------------------------
# PATHS
# -----------------------------
DATASET_JSON     = "data/dataset_3d.json"
MODEL_3D_PATH    = "models/segmentation_3d/unet3d_best_raw.pth"
MODEL_2D_PATH    = "models/smooth_0.8157/unet_best.pth"
PRED_3D_DIR      = "outputs/segmentation_3d_predictions"
OUTPUT_DIR       = "outputs/ensemble_2d_3d"
ENSEMBLE_3D_DIR  = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(ENSEMBLE_3D_DIR, exist_ok=True)

# Weight combinations to sweep on validation set
# Format: (weight_3d, weight_2d) — must sum to 1.0
WEIGHT_COMBOS = [
    (1.00, 0.00),   # 3D alone (baseline)
    (0.80, 0.20),
    (0.75, 0.25),
    (0.70, 0.30),
    (0.65, 0.35),
    (0.60, 0.40),
    (0.55, 0.45),
    (0.50, 0.50),
    (0.00, 1.00),   # 2D alone (your previous baseline)
]

FINAL_THRESHOLD = 0.4    # applied after weighted average

assert torch.cuda.is_available(), "CUDA GPU not found. This script requires a GPU."
device = torch.device("cuda")
torch.cuda.set_device(0)


# -----------------------------
# LOAD MODELS
# -----------------------------
print("\nLoading 3D model...")
model_3d = get_3d_model(device)
model_3d.load_state_dict(
    torch.load(MODEL_3D_PATH, map_location=device, weights_only=True)
)
model_3d.eval()

print("Loading 2D model...")
model_2d = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = None,
    in_channels     = 3,
    classes         = 1,
    activation      = None,
).to(device)
model_2d.load_state_dict(
    torch.load(MODEL_2D_PATH, map_location=device, weights_only=True)
)
model_2d.eval()
print("Both models loaded.")


# -----------------------------
# 2D INFERENCE HELPER
# Runs the 2D model slice-by-slice on a 3D volume
# and returns a stacked probability volume (D, H, W)
# -----------------------------
def get_2d_prob_volume(volume_np, target_shape):
    """
    volume_np    : (3, D, H, W) numpy float32 — the 3-channel MRI volume
    target_shape : (D, H, W) — spatial shape to match the 3D prediction

    Returns: prob_volume (D, H, W) float32
    """
    _, D, H, W = volume_np.shape
    prob_slices = []

    for i in range(D):
        # Extract slice (3, H, W)
        slice_3ch = volume_np[:, i, :, :]     # (3, H, W)
        # Convert to uint8 for consistency with your 2D preprocessing
        slice_uint8 = (slice_3ch * 255).astype(np.uint8)
        # Resize to 256×256 (your 2D model's expected input)
        channels_resized = []
        for c in range(3):
            ch = cv2.resize(slice_uint8[c], (256, 256))
            channels_resized.append(ch)
        img_256 = np.stack(channels_resized, axis=2)  # (256, 256, 3)

        # Normalize
        img_f = img_256.astype(np.float32) / 255.0
        img_t = torch.tensor(
            np.transpose(img_f, (2, 0, 1))
        ).unsqueeze(0).float().to(device)  # (1, 3, 256, 256)

        with torch.no_grad():
            logit = model_2d(img_t)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

        # Resize back to original H×W
        prob_orig = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        prob_slices.append(prob_orig)

    prob_vol = np.stack(prob_slices, axis=0)  # (D, H, W)

    # Resize to match 3D prediction spatial shape if needed
    if prob_vol.shape != target_shape:
        prob_vol_resized = np.zeros(target_shape, dtype=np.float32)
        for i in range(target_shape[0]):
            src_idx = int(i * D / target_shape[0])
            prob_vol_resized[i] = cv2.resize(
                prob_vol[min(src_idx, D-1)],
                (target_shape[2], target_shape[1]),
                interpolation=cv2.INTER_LINEAR
            )
        return prob_vol_resized

    return prob_vol.astype(np.float32)


# -----------------------------
# DICE HELPER
# -----------------------------
def dice_3d(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter) / (pred.sum() + gt.sum() + 1e-8)


# -----------------------------
# WEIGHT SWEEP ON VALIDATION SET
# -----------------------------
with open(DATASET_JSON) as f:
    ds_json = json.load(f)

val_data  = _expand_paths(ds_json["validation"])
val_ds    = MonaiDataset(data=val_data, transform=get_val_transforms())
val_loader = MonaiLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

print(f"\nSweeping weights on {len(val_data)} validation patients...")
print(f"{'w_3d':>6} | {'w_2d':>6} | {'Dice':>8} | {'Missed':>7}")
print("-" * 38)

sweep_results = []

for w3d, w2d in WEIGHT_COMBOS:

    val_dices = []
    missed    = 0

    for idx, batch in enumerate(val_loader):
        images = batch["image"]     # (1, 3, D, H, W)
        labels = batch["label"]

        gt_np = (labels.squeeze().numpy() > 0).astype(bool)

        # ---- 3D probability ----
        with torch.no_grad():
            with autocast():
                logits_3d = sliding_window_inference(
                    inputs=images.to(device),
                    roi_size=PATCH_SIZE,
                    sw_batch_size=4,
                    predictor=model_3d,
                    overlap=0.5,
                    mode="gaussian",
                )
        prob_3d = torch.sigmoid(logits_3d).squeeze().cpu().numpy()

        # ---- 2D probability (slice-by-slice) ----
        vol_np  = images.squeeze(0).numpy()  # (3, D, H, W)
        prob_2d = get_2d_prob_volume(vol_np, prob_3d.shape)

        # ---- Weighted ensemble ----
        prob_ensemble = w3d * prob_3d + w2d * prob_2d
        pred_bin      = (prob_ensemble > FINAL_THRESHOLD).astype(bool)

        if gt_np.sum() > 0:
            d = dice_3d(pred_bin, gt_np)
            val_dices.append(d)
            if pred_bin.sum() == 0:
                missed += 1

    mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
    sweep_results.append((w3d, w2d, mean_dice, missed))
    print(f"{w3d:>6.2f} | {w2d:>6.2f} | {mean_dice:>8.4f} | {missed:>7}")

# Find best weights
best = max(sweep_results, key=lambda x: x[2])
print(f"\nBest weights: 3D={best[0]:.2f}  2D={best[1]:.2f}  "
      f"Val Dice={best[2]:.4f}  Missed={best[3]}")

BEST_W3D = best[0]
BEST_W2D = best[1]


# -----------------------------
# RUN FINAL ENSEMBLE ON TEST SET
# -----------------------------
print(f"\nRunning final ensemble on test set "
      f"(w3d={BEST_W3D:.2f}, w2d={BEST_W2D:.2f})...")

test_data  = _expand_paths(ds_json["test"])
test_ds    = MonaiDataset(data=test_data, transform=get_val_transforms())
test_loader = MonaiLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

test_dices = []
missed_test = 0

for idx, batch in enumerate(tqdm(test_loader, desc="Test ensemble")):
    images = batch["image"]
    labels = batch["label"]

    patient_id = os.path.basename(os.path.dirname(test_data[idx]["label"]))
    gt_nii     = nib.load(test_data[idx]["label"])
    affine     = gt_nii.affine
    header     = gt_nii.header
    gt_np      = (gt_nii.get_fdata() > 0).astype(bool)

    # 3D prob
    with torch.no_grad():
        with autocast():
            logits_3d = sliding_window_inference(
                inputs=images.to(device),
                roi_size=PATCH_SIZE,
                sw_batch_size=4,
                predictor=model_3d,
                overlap=0.5,
                mode="gaussian",
            )
    prob_3d = torch.sigmoid(logits_3d).squeeze().cpu().numpy()

    # 2D prob
    vol_np  = images.squeeze(0).numpy()
    prob_2d = get_2d_prob_volume(vol_np, prob_3d.shape)

    # Ensemble
    prob_ensemble = BEST_W3D * prob_3d + BEST_W2D * prob_2d
    pred_bin      = (prob_ensemble > FINAL_THRESHOLD).astype(np.uint8)

    # Save ensemble mask as NIfTI
    out_name = f"patient{patient_id}_ensemble_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(pred_bin, affine, header),
        os.path.join(ENSEMBLE_3D_DIR, out_name)
    )

    if gt_np.sum() > 0:
        d = dice_3d(pred_bin.astype(bool), gt_np)
        test_dices.append(d)
        if pred_bin.sum() == 0:
            missed_test += 1


# -----------------------------
# FINAL REPORT
# -----------------------------
print()
print("=" * 58)
print("  ENSEMBLE RESULTS — TEST SET")
print("=" * 58)
print(f"  Weights         : 3D={BEST_W3D:.2f}  2D={BEST_W2D:.2f}")
print(f"  Threshold       : {FINAL_THRESHOLD}")
print()
print(f"  2D only Dice    : 0.7501  (your previous best)")
if test_dices:
    mean_3d_alone = float(np.mean([
        dice_3d(
            (nib.load(os.path.join(
                PRED_3D_DIR,
                f"patient{os.path.basename(os.path.dirname(e['label']))}_pred_mask.nii.gz"
            )).get_fdata() > 0),
            (nib.load(e["label"]).get_fdata() > 0)
        )
        for e in test_data
        if os.path.exists(os.path.join(
            PRED_3D_DIR,
            f"patient{os.path.basename(os.path.dirname(e['label']))}_pred_mask.nii.gz"
        ))
    ]))
    print(f"  3D only Dice    : {mean_3d_alone:.4f}")
    print(f"  Ensemble Dice   : {np.mean(test_dices):.4f}  ← final result")
    print(f"  Best patient    : {np.max(test_dices):.4f}")
    print(f"  Missed tumours  : {missed_test}")
    det = (1 - missed_test / len(test_dices)) * 100 if test_dices else 0
    print(f"  Detection rate  : {det:.1f}%")
print()
print(f"  Ensemble masks saved to: {ENSEMBLE_3D_DIR}")
print("=" * 58)
