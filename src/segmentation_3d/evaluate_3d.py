"""
evaluate_3d.py
==============

Standalone evaluation of MONAI 3D UNet predictions against ground truth.

Fix applied:
  Shape mismatch — pred_arr from .nii.gz has the padded inference shape
  (e.g. 202×96×130) while gt_arr from .npy has the original volume shape
  (e.g. 202×91×130).  The GT loaded through get_val_transforms() is padded
  to PATCH_SIZE before inference, but the label .npy on disk is unpadded.
  Solution: load the raw .npy label (unpadded), and crop the prediction
  to match using the GT shape — center-crop on any dimension that differs.

Run AFTER predict_3d.py has completed.

Usage:
    python src/segmentation_3d/evaluate_3d.py
"""

import os
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(__file__))
from data_3d import _build_data_list

# ------------------------------------------------------------------ #
#  PATHS                                                               #
# ------------------------------------------------------------------ #

PRED_DIR          = "outputs/segmentation_3d_predictions"
PREPROCESSED_ROOT = "data/patients_combined"
REPORT_FILE       = "outputs/segmentation_3d_evaluation.csv"
BASELINE_2D_DICE  = 0.7501

os.makedirs("outputs", exist_ok=True)


# ------------------------------------------------------------------ #
#  SHAPE ALIGNMENT                                                     #
#  Prediction comes from sliding_window_inference on a SpatialPadded  #
#  volume — may be larger than the raw .npy label on disk.            #
#  Center-crop the prediction to the GT shape on any axis that differs.#
# ------------------------------------------------------------------ #

def center_crop_to(pred_arr, target_shape):
    """
    Center-crop pred_arr to target_shape along each axis.
    Works for 3D arrays (D, H, W).
    If pred is smaller than target on any axis, returns pred unchanged
    for that axis (shouldn't happen — pred is always >= GT after padding).
    """
    result = pred_arr
    for axis in range(3):
        diff = result.shape[axis] - target_shape[axis]
        if diff > 0:
            start = diff // 2
            slices = [slice(None)] * 3
            slices[axis] = slice(start, start + target_shape[axis])
            result = result[tuple(slices)]
    return result


# ------------------------------------------------------------------ #
#  METRICS                                                             #
# ------------------------------------------------------------------ #

def dice_3d(pred, gt):
    pred  = pred.astype(bool)
    gt    = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter) / (pred.sum() + gt.sum() + 1e-8)


def iou_3d(pred, gt):
    pred  = pred.astype(bool)
    gt    = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / (union + 1e-8)


# ------------------------------------------------------------------ #
#  LOAD TEST DATA                                                      #
# ------------------------------------------------------------------ #

test_list = _build_data_list(PREPROCESSED_ROOT, "test")

print()
print("=" * 55)
print("  MONAI 3D UNet Evaluation — Test Set")
print("=" * 55)
print(f"  Predictions : {PRED_DIR}")
print(f"  GT labels   : {PREPROCESSED_ROOT}/test/")
print(f"  Patients    : {len(test_list)}")
print()

all_dice  = []
all_iou   = []
records   = []
missed    = 0
total_tum = 0
shape_mismatches = 0

# ------------------------------------------------------------------ #
#  MAIN LOOP                                                           #
# ------------------------------------------------------------------ #

for entry in tqdm(test_list, desc="Evaluating", ncols=80):
    patient_id = entry["patient_id"]

    label_path = entry["label"]   # .npy  (1, D, H, W)
    pred_path  = os.path.join(
        PRED_DIR, f"patient{patient_id}_pred_mask.nii.gz")

    if not os.path.exists(pred_path):
        print(f"  [MISSING] patient {patient_id} — run predict_3d.py first")
        continue

    # ---- GT: load raw .npy, squeeze channel dim → (D, H, W) ----
    gt_arr = np.load(label_path)[0]
    gt_arr = (gt_arr > 0).astype(np.uint8)
    gt_shape = gt_arr.shape   # original unpadded shape

    # ---- Prediction: load .nii.gz → (D, H, W) ----
    pred_arr = nib.load(pred_path).get_fdata()
    pred_arr = (pred_arr > 0).astype(np.uint8)

    # ---- Shape alignment ----
    if pred_arr.shape != gt_shape:
        shape_mismatches += 1
        pred_arr = center_crop_to(pred_arr, gt_shape)

    # Verify shapes now match
    if pred_arr.shape != gt_shape:
        print(f"  [SHAPE ERROR] patient {patient_id}: "
              f"pred {pred_arr.shape} vs gt {gt_shape} — skipping")
        continue

    # ---- Metrics ----
    d   = dice_3d(pred_arr, gt_arr)
    iou = iou_3d(pred_arr, gt_arr)

    has_tumor = int(gt_arr.sum() > 0)
    has_pred  = int(pred_arr.sum() > 0)

    if has_tumor:
        total_tum += 1
        all_dice.append(d)
        all_iou.append(iou)
        if not has_pred:
            missed += 1

    records.append([
        patient_id,
        round(d, 6),
        round(iou, 6),
        has_tumor,
        has_pred,
    ])

# ------------------------------------------------------------------ #
#  RESULTS                                                             #
# ------------------------------------------------------------------ #

det_rate  = 100 * (total_tum - missed) / total_tum if total_tum else 0.0
mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
mean_iou  = float(np.mean(all_iou))  if all_iou  else 0.0

print()
print("=" * 55)
print("  MONAI 3D UNet Test Set Results")
print("=" * 55)
print(f"  Patients evaluated : {len(records)}")
print(f"  Tumour patients    : {total_tum}")
print(f"  Missed tumours     : {missed}  ({100 - det_rate:.1f}%)")
print(f"  Detection rate     : {det_rate:.1f}%")
if shape_mismatches > 0:
    print(f"  Shape mismatches   : {shape_mismatches} (auto-cropped, now resolved)")
print()
print(f"  Mean Dice (tumour) : {mean_dice:.4f}")
print(f"  Mean IoU  (tumour) : {mean_iou:.4f}")
print(f"  Best Dice          : {max(all_dice):.4f}" if all_dice else "  Best Dice  : N/A")
print(f"  Worst Dice         : {min(all_dice):.4f}" if all_dice else "  Worst Dice : N/A")
print()
print(f"  2D baseline Dice   : {BASELINE_2D_DICE}")
print(f"  MONAI vs 2D base   : {mean_dice - BASELINE_2D_DICE:+.4f}")
print()

# ------------------------------------------------------------------ #
#  SAVE CSV                                                            #
# ------------------------------------------------------------------ #

with open(REPORT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patient_id", "dice", "iou", "has_tumour", "predicted"])
    writer.writerows(records)

print(f"  Report saved : {REPORT_FILE}")
print()
print("  Next step: python src/segmentation_3d/evaluate_dynunet.py")
print("=" * 55)