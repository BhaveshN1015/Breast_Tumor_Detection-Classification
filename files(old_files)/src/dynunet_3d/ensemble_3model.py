"""
ensemble_3model.py
==================

Combination C ensemble: MONAI ResNet UNet 3D + DynUNet 3D + 2D ResNet34.

Fixes applied:
  - MODEL_2D_PATH now auto-searches models/ for any unet_best.pth — no hardcoded name
  - 2D slice normalisation fixed: z-score → [0,1] clip instead of *255 (clipped negatives)
  - Prob maps precomputed ONCE per patient, weight sweep runs over cached arrays (20x faster)
  - Weight+threshold co-swept: 3 thresholds × 20 weight combos = 60 combos total
  - Best threshold saved alongside best weights and applied on test set
  - Keeps full HD95, IoU, volume metrics on test set
  - All previous paths + structures preserved exactly

Usage:
    python src/segmentation_3d/ensemble_3model.py

Requirements:
  - train_3d.py done      → models/segmentation_3d/unet3d_best_raw.pth
  - train_dynunet.py done → models/dynunet_3d/dynunet_best_raw.pth
  - 2D model anywhere under models/ named unet_best.pth
  - predict_3d.py done    → outputs/segmentation_3d_predictions/
  - predict_dynunet.py done → outputs/dynunet_predictions/
"""

import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import cv2
from torch.amp import autocast
from tqdm import tqdm
from scipy.ndimage import (binary_erosion, generate_binary_structure,
                            binary_closing, label as scipy_label)
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from monai.inferers import sliding_window_inference

sys.path.append(os.path.dirname(__file__))

from data_3d       import NpyDataset, get_val_transforms, PATCH_SIZE, _build_data_list
from model_3d      import get_model as get_monai_model
from model_dynunet import get_dynunet

# ------------------------------------------------------------------ #
#  PATHS                                                               #
# ------------------------------------------------------------------ #

PREPROCESSED_ROOT = "data/patients_combined"
MODEL_MONAI_PATH  = "models/segmentation_3d/unet3d_best_raw.pth"
MODEL_DYN_PATH    = "models/dynunet_3d/dynunet_best_raw.pth"
PRED_MONAI_DIR    = "outputs/segmentation_3d_predictions"
PRED_DYN_DIR      = "outputs/dynunet_predictions"
OUTPUT_DIR        = "outputs/ensemble_3model"
ENSEMBLE_PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
REPORT_FILE       = os.path.join(OUTPUT_DIR, "ensemble_3model_evaluation.csv")

os.makedirs(ENSEMBLE_PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,        exist_ok=True)

# ------------------------------------------------------------------ #
#  FIX 1 — AUTO-FIND 2D MODEL PATH                                    #
#  Searches the entire models/ tree for unet_best.pth so the script  #
#  never fails because a folder was renamed (e.g. smooth_0.8157).    #
# ------------------------------------------------------------------ #

def _find_2d_model():
    """Return the first unet_best.pth found under models/."""
    candidates = glob.glob(os.path.join("models", "**", "unet_best.pth"),
                           recursive=True)
    if candidates:
        # Sort so the deepest/most-specific path is preferred
        candidates.sort(key=lambda p: (len(p.split(os.sep)), p))
        return candidates[0]
    return None

MODEL_2D_PATH = _find_2d_model()

if MODEL_2D_PATH is None:
    raise FileNotFoundError(
        "\n\n  [ERROR] Could not find unet_best.pth anywhere under models/\n"
        "  Expected location example: models/smooth_0.8157/unet_best.pth\n"
        "  Make sure your 2D classifier training has completed and the\n"
        "  checkpoint is saved under the models/ directory.\n"
    )

print(f"  Found 2D model → {MODEL_2D_PATH}")

# ------------------------------------------------------------------ #
#  WEIGHT COMBOS + THRESHOLD SWEEP                                     #
#  FIX 2 — Sweep threshold alongside weights (was fixed at 0.2).     #
#  Three thresholds × 20 weight combos = 60 total configurations.    #
# ------------------------------------------------------------------ #

WEIGHT_COMBOS = [
    (1.00, 0.00, 0.00),
    (0.00, 1.00, 0.00),
    (0.00, 0.00, 1.00),
    (0.50, 0.50, 0.00),
    (0.45, 0.45, 0.10),
    (0.40, 0.40, 0.20),
    (0.38, 0.37, 0.25),
    (0.35, 0.35, 0.30),
    (0.40, 0.35, 0.25),
    (0.35, 0.40, 0.25),
    (0.45, 0.30, 0.25),
    (0.30, 0.45, 0.25),
    (0.40, 0.30, 0.30),
    (0.30, 0.40, 0.30),
    (0.35, 0.30, 0.35),
    (0.30, 0.35, 0.35),
    (0.50, 0.30, 0.20),
    (0.30, 0.50, 0.20),
    (0.60, 0.25, 0.15),
    (0.25, 0.60, 0.15),
]

# Threshold sweep — both models trained with optimal thresh 0.7;
# combined map needs its own sweep because the linear blend shifts the distribution.
SWEEP_THRESHOLDS = [0.2, 0.35, 0.5]
FINAL_THRESHOLD  = 0.35   # default; overwritten by sweep result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if device.type == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ------------------------------------------------------------------ #
#  LOAD MODELS                                                         #
# ------------------------------------------------------------------ #

print("\nLoading MONAI ResNet UNet 3D...")
model_monai = get_monai_model(device)
model_monai.load_state_dict(
    torch.load(MODEL_MONAI_PATH, map_location=device, weights_only=True))
model_monai.eval()

print("Loading DynUNet 3D...")
model_dyn = get_dynunet(device)
model_dyn.load_state_dict(
    torch.load(MODEL_DYN_PATH, map_location=device, weights_only=True))
model_dyn.eval()

print(f"Loading 2D ResNet34 U-Net from {MODEL_2D_PATH}...")
model_2d = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=1, activation=None,
).to(device)
model_2d.load_state_dict(
    torch.load(MODEL_2D_PATH, map_location=device, weights_only=True))
model_2d.eval()
print("All three models loaded.\n")


# ------------------------------------------------------------------ #
#  METRIC HELPERS                                                      #
# ------------------------------------------------------------------ #

def dice_3d(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    return float(2 * np.logical_and(pred, gt).sum()) / (pred.sum() + gt.sum() + 1e-8)

def iou_3d(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    return float(np.logical_and(pred, gt).sum()) / (np.logical_or(pred, gt).sum() + 1e-8)

def hausdorff_95(pred, gt, voxel_spacing=(1.5, 1.5, 1.5)):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    struct       = generate_binary_structure(3, 1)
    pred_surface = pred ^ binary_erosion(pred, struct)
    gt_surface   = gt   ^ binary_erosion(gt,   struct)
    pred_pts     = np.argwhere(pred_surface).astype(float) * np.array(voxel_spacing)
    gt_pts       = np.argwhere(gt_surface).astype(float)   * np.array(voxel_spacing)
    MAX = 10000
    if len(pred_pts) > MAX:
        pred_pts = pred_pts[np.random.choice(len(pred_pts), MAX, replace=False)]
    if len(gt_pts) > MAX:
        gt_pts   = gt_pts[np.random.choice(len(gt_pts),   MAX, replace=False)]
    def directed(src, tgt):
        mins = []
        for i in range(0, len(src), 500):
            s = src[i:i+500]
            mins.append(np.sqrt(((s[:, None, :] - tgt[None, :, :])**2).sum(2)).min(1))
        return np.concatenate(mins)
    return float(np.percentile(
        np.concatenate([directed(pred_pts, gt_pts), directed(gt_pts, pred_pts)]), 95))

def tumour_volume_mm3(mask, voxel_spacing=(1.5, 1.5, 1.5)):
    return float(mask.sum()) * float(np.prod(voxel_spacing))

def post_process_3d(binary_mask, min_size_voxels=50):
    struct      = generate_binary_structure(3, 1)
    closed      = binary_closing(binary_mask, structure=struct, iterations=2)
    labeled_arr, n = scipy_label(closed)
    if n == 0:
        return closed.astype(np.uint8)
    cleaned = np.zeros_like(closed, dtype=np.uint8)
    for i in range(1, n + 1):
        if (labeled_arr == i).sum() >= min_size_voxels:
            cleaned[labeled_arr == i] = 1
    return cleaned


# ------------------------------------------------------------------ #
#  3D INFERENCE HELPERS                                                #
# ------------------------------------------------------------------ #

def get_prob_monai(image_tensor):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(
                inputs=image_tensor, roi_size=PATCH_SIZE,
                sw_batch_size=4, predictor=model_monai,
                overlap=0.5, mode="gaussian")
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


def get_prob_dynunet(image_tensor):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(
                inputs=image_tensor, roi_size=PATCH_SIZE,
                sw_batch_size=4, predictor=model_dyn,
                overlap=0.5, mode="gaussian")
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


def get_prob_2d(volume_np, target_shape):
    """
    Run 2D ResNet34 slice-by-slice on the 3D volume.

    FIX 3 — z-score normalisation: the 3D volumes are z-score normalised
    (mean≈0, std≈1). The old code multiplied by 255 and cast to uint8,
    which zeroed all negative values (a large fraction of each slice).
    The fix clips to the [-3, 6] z-score range (covers 99%+ of voxels)
    and rescales linearly to [0, 1] before feeding the 2D model.

    volume_np    : (3, D, H, W) float32 z-score normalised
    target_shape : (D', H', W') — shape of the 3D prob map to match
    Returns      : prob_volume (D', H', W') float32
    """
    _, D, H, W = volume_np.shape
    prob_slices = []

    for i in range(D):
        slice_3ch = volume_np[:, i, :, :]   # (3, H, W)

        # Rescale z-score to [0,1] preserving sign information
        slice_clipped = np.clip(slice_3ch, -3.0, 6.0)
        slice_norm    = (slice_clipped - (-3.0)) / (6.0 - (-3.0))   # [0,1]
        slice_uint8   = (slice_norm * 255).astype(np.uint8)          # [0,255]

        channels_resized = []
        for c in range(3):
            channels_resized.append(cv2.resize(slice_uint8[c], (256, 256)))

        img_f = np.stack(channels_resized, axis=2).astype(np.float32) / 255.0
        img_t = torch.tensor(
            np.transpose(img_f, (2, 0, 1))
        ).unsqueeze(0).float().to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model_2d(img_t)).squeeze().cpu().numpy()

        prob_slices.append(cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR))

    prob_vol = np.stack(prob_slices, axis=0)   # (D, H, W)

    if prob_vol.shape != target_shape:
        resized = np.zeros(target_shape, dtype=np.float32)
        for i in range(target_shape[0]):
            src_idx = int(i * D / target_shape[0])
            resized[i] = cv2.resize(
                prob_vol[min(src_idx, D - 1)],
                (target_shape[2], target_shape[1]),
                interpolation=cv2.INTER_LINEAR)
        return resized

    return prob_vol.astype(np.float32)


# ------------------------------------------------------------------ #
#  LOAD DATASETS                                                       #
# ------------------------------------------------------------------ #

val_list  = _build_data_list(PREPROCESSED_ROOT, "val")
test_list = _build_data_list(PREPROCESSED_ROOT, "test")

val_ds    = NpyDataset(val_list,  transform=get_val_transforms())
test_ds   = NpyDataset(test_list, transform=get_val_transforms())

val_loader  = DataLoader(val_ds,  batch_size=1, shuffle=False,
                         num_workers=0, pin_memory=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=0, pin_memory=False)


# ------------------------------------------------------------------ #
#  FIX 4 — PRECOMPUTE PROBABILITY MAPS ONCE FOR ALL VAL PATIENTS      #
#  Original code ran 3 full model inferences per patient PER WEIGHT   #
#  COMBO → 20 combos × 34 patients = 680 inference passes.           #
#  Precomputing reduces this to 34 passes — a 20× speedup.           #
# ------------------------------------------------------------------ #

print(f"Precomputing probability maps for {len(val_list)} val patients...")
print("  (MONAI + DynUNet + 2D — done once, reused across all weight combos)\n")

val_cache = []   # list of (prob_monai, prob_dyn, prob_2d, gt_np)

for idx, batch in enumerate(tqdm(val_loader, desc="  Precomputing val", ncols=90)):
    images = batch["image"]
    labels = batch["label"]
    gt_np  = (labels.squeeze().numpy() > 0).astype(bool)

    pm = get_prob_monai(images.to(device))
    pd = get_prob_dynunet(images.to(device))
    p2 = get_prob_2d(images.squeeze(0).numpy(), pm.shape)

    val_cache.append((pm, pd, p2, gt_np))

print(f"\n  Precomputation done — {len(val_cache)} patients cached.\n")


# ------------------------------------------------------------------ #
#  WEIGHT + THRESHOLD SWEEP ON VALIDATION SET                         #
#  FIX 5 — Sweep threshold alongside weights.                         #
#  MONAI optimal threshold is 0.7; combined map has a different       #
#  optimal threshold because the linear blend shifts the distribution. #
# ------------------------------------------------------------------ #

n_combos = len(WEIGHT_COMBOS) * len(SWEEP_THRESHOLDS)
print(f"Sweeping {len(WEIGHT_COMBOS)} weight combos × {len(SWEEP_THRESHOLDS)} thresholds"
      f" = {n_combos} total configurations on {len(val_list)} val patients...")
print(f"{'w_monai':>8} | {'w_dyn':>6} | {'w_2d':>6} | {'thresh':>7} | "
      f"{'Val Dice':>10} | {'Missed':>7}")
print("-" * 58)

sweep_results = []

for w_monai, w_dyn, w_2d in WEIGHT_COMBOS:
    for thresh in SWEEP_THRESHOLDS:
        val_dices = []
        missed    = 0

        for pm, pd, p2, gt_np in val_cache:
            prob_ens = w_monai * pm + w_dyn * pd + w_2d * p2
            pred_bin = (prob_ens > thresh).astype(bool)

            if gt_np.sum() > 0:
                d = dice_3d(pred_bin, gt_np)
                val_dices.append(d)
                if pred_bin.sum() == 0:
                    missed += 1

        mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
        sweep_results.append((w_monai, w_dyn, w_2d, thresh, mean_dice, missed))

        print(f"{w_monai:>8.2f} | {w_dyn:>6.2f} | {w_2d:>6.2f} | "
              f"{thresh:>7.2f} | {mean_dice:>10.4f} | {missed:>7}")

best = max(sweep_results, key=lambda x: x[4])
BEST_W_MONAI, BEST_W_DYN, BEST_W_2D = best[0], best[1], best[2]
FINAL_THRESHOLD = best[3]

print()
print("=" * 58)
print(f"Best combo : MONAI={BEST_W_MONAI:.2f}  DynUNet={BEST_W_DYN:.2f}  2D={BEST_W_2D:.2f}")
print(f"Threshold  : {FINAL_THRESHOLD:.2f}")
print(f"Val Dice   : {best[4]:.4f}  |  Missed: {best[5]}")
print("=" * 58)


# ------------------------------------------------------------------ #
#  FINAL ENSEMBLE ON TEST SET                                         #
# ------------------------------------------------------------------ #

print(f"\nRunning final ensemble on {len(test_list)} test patients...")
print(f"  Weights  : MONAI={BEST_W_MONAI:.2f}  DynUNet={BEST_W_DYN:.2f}  2D={BEST_W_2D:.2f}")
print(f"  Threshold: {FINAL_THRESHOLD:.2f}\n")

records      = []
test_dices   = []
test_ious    = []
test_hd95s   = []
missed_test  = 0
total_tumors = 0
total_tp = total_fp = total_fn = total_tn = 0.0

test_bar = tqdm(
    test_loader, desc="  Test ensemble", leave=True, ncols=110,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
)

for idx, batch in enumerate(test_bar):
    images     = batch["image"]
    labels     = batch["label"]
    patient_id = test_list[idx]["patient_id"]
    gt_np      = (labels.squeeze().numpy() > 0).astype(bool)

    prob_monai = get_prob_monai(images.to(device))
    prob_dyn   = get_prob_dynunet(images.to(device))
    prob_2d    = get_prob_2d(images.squeeze(0).numpy(), prob_monai.shape)

    prob_ens  = BEST_W_MONAI * prob_monai + BEST_W_DYN * prob_dyn + BEST_W_2D * prob_2d
    pred_bin  = post_process_3d((prob_ens > FINAL_THRESHOLD).astype(np.uint8))
    pred_bool = pred_bin.astype(bool)

    nib.save(nib.Nifti1Image(pred_bin, np.eye(4)),
             os.path.join(ENSEMBLE_PRED_DIR,
                          f"patient{patient_id}_ensemble_mask.nii.gz"))

    d   = dice_3d(pred_bool, gt_np)
    iou = iou_3d(pred_bool, gt_np)
    hd  = hausdorff_95(pred_bool, gt_np)
    gt_vol   = tumour_volume_mm3(gt_np)
    pred_vol = tumour_volume_mm3(pred_bool)

    p = pred_bool.astype(float); t = gt_np.astype(float)
    tp = (p * t).sum();      total_tp += tp
    fp = (p * (1 - t)).sum(); total_fp += fp
    fn = ((1 - p) * t).sum(); total_fn += fn
    tn = ((1 - p) * (1 - t)).sum(); total_tn += tn

    if gt_np.sum() > 0:
        total_tumors += 1
        test_dices.append(d)
        test_ious.append(iou)
        if hd is not None:
            test_hd95s.append(hd)
        if pred_bool.sum() == 0:
            missed_test += 1

    records.append([patient_id, round(d, 6), round(iou, 6),
                    round(hd, 2) if hd is not None else "N/A",
                    int(gt_np.sum() > 0), int(pred_bool.sum() > 0),
                    round(gt_vol, 1), round(pred_vol, 1)])

    test_bar.set_postfix(
        Dice=f"{d:.4f}", IoU=f"{iou:.4f}",
        Prec=f"{tp/(tp+fp+1e-6):.4f}", Rec=f"{tp/(tp+fn+1e-6):.4f}",
        Acc=f"{(tp+tn)/(tp+tn+fp+fn+1e-6):.4f}",
    )


# ------------------------------------------------------------------ #
#  SAVE CSV + FINAL REPORT                                            #
# ------------------------------------------------------------------ #

with open(REPORT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "dice", "iou", "hd95_mm", "has_tumour",
                     "pred_has_tumour", "gt_volume_mm3", "pred_volume_mm3"])
    writer.writerows(records)
print(f"\nPer-patient report saved to: {REPORT_FILE}")

global_prec = total_tp / (total_tp + total_fp + 1e-6)
global_rec  = total_tp / (total_tp + total_fn + 1e-6)
global_acc  = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-6)
miss_pct    = missed_test / total_tumors * 100 if total_tumors else 0

print()
print("=" * 60)
print("  COMBINATION C ENSEMBLE — FINAL TEST SET RESULTS")
print("=" * 60)
print(f"  Weights  : MONAI={BEST_W_MONAI:.2f}  DynUNet={BEST_W_DYN:.2f}  2D={BEST_W_2D:.2f}")
print(f"  Threshold: {FINAL_THRESHOLD:.2f}")
print()
if test_dices:
    print(f"  Tumour Dice      : {np.mean(test_dices):.4f}  ← main metric")
    print(f"  Tumour IoU       : {np.mean(test_ious):.4f}")
    if test_hd95s:
        print(f"  Tumour HD95      : {np.mean(test_hd95s):.2f} mm")
    print(f"  Best Dice        : {np.max(test_dices):.4f}")
    print(f"  Worst Dice       : {np.min(test_dices):.4f}")
print()
print(f"  Total test patients : {len(records)}")
print(f"  Tumour patients     : {total_tumors}")
print(f"  Missed tumours      : {missed_test}  ({miss_pct:.1f}%)")
print(f"  Detection rate      : {100 - miss_pct:.1f}%")
print()
print(f"  Precision  : {global_prec:.4f}")
print(f"  Recall     : {global_rec:.4f}")
print(f"  Accuracy   : {global_acc:.4f}")
print()
print(f"  2D baseline Dice     : 0.7501")
if test_dices:
    ens_dice = np.mean(test_dices)
    print(f"  Ensemble Dice        : {ens_dice:.4f}")
    print(f"  Gain vs 2D baseline  : {ens_dice - 0.7501:+.4f}")
print()
print(f"  Ensemble masks saved to: {ENSEMBLE_PRED_DIR}")
print("=" * 60)