import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

PRED_DIR    = "outputs/segmentation_predictions"
GT_DIR      = "data/Segmentation_dataset/test/masks"
REPORT_FILE = "outputs/segmentation_evaluation_report.csv"

os.makedirs("outputs", exist_ok=True)


def dice_score(pred, gt):
    pred = pred > 0
    gt   = gt   > 0
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)


def iou_score(pred, gt):
    pred = pred > 0
    gt   = gt   > 0
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    return intersection / (union + 1e-8)


# -------------------------------------------------------
# Separate tracking for:
#   - all slices (overall honest metric)
#   - tumor slices only (segmentation quality metric)
#   - small tumor slices (early detection metric — new)
# -------------------------------------------------------
all_dice   = []
all_iou    = []

tumor_dice = []
tumor_iou  = []

# Small tumor = tumor present but occupies < 2% of image pixels
# 2% of 256x256 = 1310 pixels
SMALL_TUMOR_THRESHOLD = int(256 * 256 * 0.02)
small_tumor_dice = []
small_tumor_iou  = []

records = []

files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".png")])
print(f"Evaluating {len(files)} predictions...")

total_count       = 0
tumor_count       = 0
small_tumor_count = 0
missed_tumors     = 0   # GT has tumor but prediction is empty


for img_name in tqdm(files):

    pred_path = os.path.join(PRED_DIR, img_name)
    mask_name = img_name.replace(".png", "_mask.png")
    gt_path   = os.path.join(GT_DIR, mask_name)

    if not os.path.exists(gt_path):
        continue

    total_count += 1

    pred = cv2.imread(pred_path, 0)
    gt   = cv2.imread(gt_path,   0)

    if pred is None or gt is None:
        continue

    pred = cv2.resize(pred, (256, 256))
    gt   = cv2.resize(gt,   (256, 256))

    d = dice_score(pred, gt)
    i = iou_score(pred, gt)

    all_dice.append(d)
    all_iou.append(i)

    has_tumor     = gt.sum() > 0
    tumor_pixels  = np.sum(gt > 0)
    pred_has_tumor = pred.sum() > 0

    if has_tumor:
        tumor_count += 1
        tumor_dice.append(d)
        tumor_iou.append(i)

        # Track missed tumors — GT has tumor but model predicted nothing
        if not pred_has_tumor:
            missed_tumors += 1

        # Track small tumor performance separately
        if tumor_pixels < SMALL_TUMOR_THRESHOLD:
            small_tumor_count += 1
            small_tumor_dice.append(d)
            small_tumor_iou.append(i)

    records.append([
        img_name,
        round(d, 6),
        round(i, 6),
        int(has_tumor),
        tumor_pixels,
        int(pred_has_tumor)
    ])


# -------------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------------
print()
print("=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)

print("\n--- Overall (All Slices) ---")
print(f"  Mean Dice : {np.mean(all_dice):.4f}")
print(f"  Mean IoU  : {np.mean(all_iou):.4f}")

print("\n--- Tumor Slices Only ---")
if tumor_dice:
    print(f"  Tumor Dice : {np.mean(tumor_dice):.4f}  ← main metric")
    print(f"  Tumor IoU  : {np.mean(tumor_iou):.4f}")
else:
    print("  No tumor slices found.")

print("\n--- Small Tumor Slices (< 2% of image) ---")
if small_tumor_dice:
    print(f"  Small Tumor Dice : {np.mean(small_tumor_dice):.4f}  ← early detection metric")
    print(f"  Small Tumor IoU  : {np.mean(small_tumor_iou):.4f}")
    print(f"  Count            : {small_tumor_count} slices")
else:
    print("  No small tumor slices found.")

print("\n--- Dataset Statistics ---")
print(f"  Total slices      : {total_count}")
print(f"  Tumor slices      : {tumor_count}")
print(f"  Non-tumor slices  : {total_count - tumor_count}")
print(f"  Small tumor slices: {small_tumor_count}")
print(f"  Missed tumors     : {missed_tumors}  ← GT has tumor, model predicted nothing")

print("\n--- Best / Worst ---")
print(f"  Best Dice  : {np.max(all_dice):.4f}")
print(f"  Worst Dice : {np.min(all_dice):.4f}")
print("=" * 55)

# -------------------------------------------------------
# SAVE CSV REPORT
# -------------------------------------------------------
with open(REPORT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "dice", "iou",
        "contains_tumor", "tumor_pixels", "prediction_has_tumor"
    ])
    writer.writerows(records)

print(f"\nDetailed report saved to: {REPORT_FILE}")