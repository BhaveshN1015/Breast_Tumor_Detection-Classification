"""
prepare_3d_offline.py
======================

One-time offline preprocessing for breast tumour 3D segmentation.

What this does:
  1. Loads P1, P2, P3, GT, and Breast_mask NIfTI files per patient
  2. Resamples everything to 1.5mm isotropic spacing
     (native: 0.379×0.379×1.70mm → output: ~226×226×136 voxels)
  3. Reorients to RAS
  4. Crops to breast mask bounding box (removes empty air)
  5. Normalises intensities per channel (percentile + z-score)
  6. Saves as .npy files — faster to load than NIfTI during training
     image.npy  → (3, D, H, W) float32   [P1, P2, P3 stacked]
     label.npy  → (1, D, H, W) float32   [GT binary mask]

Output structure:
  data/patients_preprocessed/
    train/
      1/
        image.npy
        label.npy
      2/ ...
    val/ ...
    test/ ...

After running this:
  1. Update dataset_3d.json to point to patients_preprocessed/
  2. Replace data_3d.py _base_transforms() with the simplified version
     (only LoadNpy + ConcatItems + patch sampling — no resampling needed)

Run:
  python src/segmentation_3d/prepare_3d_offline.py

Time estimate: ~3–5 minutes for 100 patients on CPU.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    EnsureTyped,
)

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

PATIENTS_ROOT = "data/patients"
OUTPUT_ROOT   = "data/patients_preprocessed"
SPLITS        = ["train", "val", "test"]

# 1.5mm isotropic — reduces 896×896×120 to ~226×226×136
# Better than 1.0mm: smaller volumes, tumor/background ratio improves,
# training is 3-4x faster, patch sampling is much more efficient.
TARGET_SPACING = (1.5, 1.5, 1.5)

# Margin around breast mask bounding box (voxels, after resampling)
BREAST_MARGIN = 5

# ------------------------------------------------------------------ #
#  TRANSFORMS — applied to image channels and label                   #
# ------------------------------------------------------------------ #

# NOTE: breast mask is loaded separately just for bbox computation,
# not as a model input channel.

resample_orient = Compose([
    LoadImaged(
        keys=["p1", "p2", "p3", "label", "breast_mask"],
        image_only=True,
        ensure_channel_first=True,
    ),
    Spacingd(
        keys=["p1", "p2", "p3", "label", "breast_mask"],
        pixdim=TARGET_SPACING,
        mode=("bilinear", "bilinear", "bilinear", "nearest", "nearest"),
    ),
    Orientationd(
        keys=["p1", "p2", "p3", "label", "breast_mask"],
        axcodes="RAS",
    ),
    ScaleIntensityRangePercentilesd(
        keys=["p1", "p2", "p3"],
        lower=0.5, upper=99.5,
        b_min=0.0, b_max=1.0,
        clip=True,
    ),
    NormalizeIntensityd(
        keys=["p1", "p2", "p3"],
        nonzero=True,
        channel_wise=True,
    ),
    EnsureTyped(
        keys=["p1", "p2", "p3", "label", "breast_mask"],
        dtype=torch.float32,
    ),
])


# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def find_nii(folder, name):
    """Find NIfTI file by base name, trying .nii then .nii.gz."""
    for ext in [".nii", ".nii.gz"]:
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    return None


def get_breast_bbox(breast_mask_np, margin=5):
    """
    Compute bounding box of breast mask with margin.
    breast_mask_np: (1, D, H, W) numpy array
    Returns: (d_min, d_max, h_min, h_max, w_min, w_max) clipped to volume
    """
    mask = breast_mask_np.squeeze()   # (D, H, W)
    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        # No breast mask found — return full volume
        D, H, W = mask.shape
        return 0, D, 0, H, 0, W

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    D, H, W = mask.shape

    d_min = max(0,     mins[0] - margin)
    d_max = min(D,     maxs[0] + margin + 1)
    h_min = max(0,     mins[1] - margin)
    h_max = min(H,     maxs[1] + margin + 1)
    w_min = max(0,     mins[2] - margin)
    w_max = min(W,     maxs[2] + margin + 1)

    return d_min, d_max, h_min, h_max, w_min, w_max


def verify_label_integrity(label_np, patient_id):
    """
    Check that tumor voxels survived preprocessing.
    Prints a warning if the label was wiped out.
    """
    tumor_voxels = int((label_np > 0).sum())
    if tumor_voxels == 0:
        print(f"  [WARNING] Patient {patient_id}: GT label has 0 tumor "
              f"voxels after preprocessing — check this patient!")
    return tumor_voxels


# ------------------------------------------------------------------ #
#  MAIN PREPROCESSING LOOP                                            #
# ------------------------------------------------------------------ #

total_processed = 0
total_skipped   = 0
shape_log       = []   # collect shapes for reporting

print("=" * 60)
print("  Offline 3D preprocessing")
print(f"  Target spacing : {TARGET_SPACING} mm")
print(f"  Breast margin  : {BREAST_MARGIN} voxels")
print(f"  Output root    : {OUTPUT_ROOT}")
print("=" * 60)

for split in SPLITS:
    split_in  = os.path.join(PATIENTS_ROOT, split)
    split_out = os.path.join(OUTPUT_ROOT,   split)

    if not os.path.exists(split_in):
        print(f"\n[SKIP] Split folder not found: {split_in}")
        continue

    patients = sorted(
        [p for p in os.listdir(split_in)
         if os.path.isdir(os.path.join(split_in, p))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    print(f"\nProcessing {split}: {len(patients)} patients")

    for pid in tqdm(patients, desc=f"  {split}"):
        pat_in  = os.path.join(split_in,  pid)
        pat_out = os.path.join(split_out, pid)
        os.makedirs(pat_out, exist_ok=True)

        # Skip if already processed
        if (os.path.exists(os.path.join(pat_out, "image.npy")) and
                os.path.exists(os.path.join(pat_out, "label.npy"))):
            total_processed += 1
            continue

        # Locate files
        p1_path  = find_nii(pat_in, "P1")
        p2_path  = find_nii(pat_in, "P2")
        p3_path  = find_nii(pat_in, "P3")
        gt_path  = find_nii(pat_in, "GT")
        bm_path  = find_nii(pat_in, "Breast_mask")

        # Breast mask is optional — fall back to full volume if missing
        use_breast_mask = bm_path is not None

        if any(p is None for p in [p1_path, p2_path, p3_path, gt_path]):
            tqdm.write(f"  [SKIP] Patient {pid}: missing P1/P2/P3/GT")
            total_skipped += 1
            continue

        # Build item dict
        if use_breast_mask:
            item = {
                "p1": p1_path, "p2": p2_path, "p3": p3_path,
                "label": gt_path, "breast_mask": bm_path,
            }
        else:
            # Duplicate GT as breast_mask placeholder so transforms work
            item = {
                "p1": p1_path, "p2": p2_path, "p3": p3_path,
                "label": gt_path, "breast_mask": gt_path,
            }
            tqdm.write(f"  [INFO] Patient {pid}: no breast mask found, "
                       f"using full volume")

        try:
            result = resample_orient(item)
        except Exception as e:
            tqdm.write(f"  [ERROR] Patient {pid}: {e}")
            total_skipped += 1
            continue

        # Extract numpy arrays — shape (1, D, H, W)
        p1_np  = result["p1"].numpy()
        p2_np  = result["p2"].numpy()
        p3_np  = result["p3"].numpy()
        gt_np  = result["label"].numpy()
        bm_np  = result["breast_mask"].numpy()

        # Crop to breast mask bounding box
        d0, d1, h0, h1, w0, w1 = get_breast_bbox(bm_np, margin=BREAST_MARGIN)

        p1_crop  = p1_np[:,  d0:d1, h0:h1, w0:w1]
        p2_crop  = p2_np[:,  d0:d1, h0:h1, w0:w1]
        p3_crop  = p3_np[:,  d0:d1, h0:h1, w0:w1]
        gt_crop  = gt_np[:,  d0:d1, h0:h1, w0:w1]

        # Stack P1+P2+P3 into (3, D, H, W)
        image_np = np.concatenate([p1_crop, p2_crop, p3_crop], axis=0)
        label_np = gt_crop   # (1, D, H, W)

        # Verify tumor survived the crop
        tumor_voxels = verify_label_integrity(label_np, pid)

        # Save as .npy — much faster to load than NIfTI during training
        np.save(os.path.join(pat_out, "image.npy"), image_np.astype(np.float32))
        np.save(os.path.join(pat_out, "label.npy"), label_np.astype(np.float32))

        shape_log.append({
            "patient" : pid,
            "split"   : split,
            "shape"   : image_np.shape,
            "tumor_vx": tumor_voxels,
            "ratio_pct": round(tumor_voxels / label_np.size * 100, 4),
        })

        total_processed += 1

# ------------------------------------------------------------------ #
#  SUMMARY REPORT                                                      #
# ------------------------------------------------------------------ #

print()
print("=" * 60)
print("  Preprocessing complete")
print("=" * 60)
print(f"  Processed : {total_processed}")
print(f"  Skipped   : {total_skipped}")

if shape_log:
    shapes = [s["shape"] for s in shape_log]
    d_vals = [s[1] for s in shapes]
    h_vals = [s[2] for s in shapes]
    w_vals = [s[3] for s in shapes]
    ratios = [s["ratio_pct"] for s in shape_log]
    tumor_voxels_list = [s["tumor_vx"] for s in shape_log]

    print(f"\n  Volume shapes after preprocessing:")
    print(f"    D range : {min(d_vals)} – {max(d_vals)} voxels")
    print(f"    H range : {min(h_vals)} – {max(h_vals)} voxels")
    print(f"    W range : {min(w_vals)} – {max(w_vals)} voxels")
    print(f"\n  Tumor statistics:")
    print(f"    Tumor voxels  : {min(tumor_voxels_list)} – {max(tumor_voxels_list)}")
    print(f"    Tumor ratio   : {min(ratios):.4f}% – {max(ratios):.4f}%")
    print(f"    Mean ratio    : {np.mean(ratios):.4f}%")

    zero_tumor = [s["patient"] for s in shape_log if s["tumor_vx"] == 0]
    if zero_tumor:
        print(f"\n  [WARNING] Patients with 0 tumor voxels after crop:")
        for p in zero_tumor:
            print(f"    Patient {p}")
        print(f"  Check these patients — their breast mask may not overlap GT.")
    else:
        print(f"\n  All {len(shape_log)} patients have tumor voxels intact.")

print(f"\n  Output: {OUTPUT_ROOT}/")
print()
print("  Next steps:")
print("  1. python src/segmentation_3d/generate_dataset_json.py")
print("     (update PATIENTS_ROOT to 'data/patients_preprocessed')")
print("  2. Replace data_3d.py _base_transforms() with simplified version")
print("  3. python src/segmentation_3d/train_3d.py")
