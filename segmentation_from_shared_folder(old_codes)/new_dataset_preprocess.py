"""
prepare_new_dataset.py
======================
Preprocesses the new ISPY-1 dataset to match the format of
data/patients_preprocessed/ exactly.

New data:  BreastDCEDL_spy1/processed_dataset/ISPY1_XXXX/
           image_acq0.nii.gz  (pre-contrast or early phase)
           image_acq1.nii.gz  (mid phase)
           image_acq2.nii.gz  (late phase)
           mask.nii.gz        (GT tumor — binary 0/1)

Output:    data/patients_preprocessed/train|val|test/ISPY1_XXXX/
           image.npy  (3, D, H, W)  float32
           label.npy  (1, D, H, W)  float32

Key differences vs old prepare_3d_offline.py:
  - No Breast_mask.nii.gz → use image-based CropForeground
  - Already 1mm isotropic → still resample to 1.5mm to match old preprocessed
  - Already RAS → skip orientation step
  - 3 acquisitions (acq0/1/2) map directly to 3 channels
"""

import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ConcatItemsd,
    EnsureTyped,
)
import torch

# ── CONFIG ────────────────────────────────────────────────────────
NEW_DATA_ROOT  = "D:\Breast_Tumor_AI_Project\data\BreastDCEDL_spy1\processed_dataset"
OUTPUT_ROOT    = "data/new_patients_preprocessed"
TARGET_SPACING = (1.5, 1.5, 1.5)    # must match old preprocessed exactly
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
# TEST_RATIO   = 0.15  (remainder)

# Use only patients with all 3 acquisitions + mask
REQUIRED_FILES = ['image_acq0.nii.gz', 'image_acq1.nii.gz',
                  'image_acq2.nii.gz', 'mask.nii.gz']
# ──────────────────────────────────────────────────────────────────


def find_complete_patients(root):
    """Return list of patient dirs that have all required files."""
    complete = []
    skipped  = []
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        missing = [f for f in REQUIRED_FILES
                   if not os.path.exists(os.path.join(pdir, f))]
        if missing:
            skipped.append((pid, missing))
        else:
            complete.append({'pid': pid, 'dir': pdir})
    print(f"Complete patients : {len(complete)}")
    print(f"Skipped (missing) : {len(skipped)}")
    return complete, skipped


def split_patients(patients, train_r=0.70, val_r=0.15, seed=42):
    """Reproducible train/val/test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n   = len(patients)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return (
        [patients[i] for i in idx[:n_train]],
        [patients[i] for i in idx[n_train:n_train+n_val]],
        [patients[i] for i in idx[n_train+n_val:]],
    )


def preprocess_patient(patient_info, out_dir, resume=True):
    """
    Full preprocessing pipeline for one ISPY-1 patient.
    Returns dict with shape and tumor stats, or None if skipped.
    """
    pid  = patient_info['pid']
    pdir = patient_info['dir']
    out  = os.path.join(out_dir, pid)
    os.makedirs(out, exist_ok=True)

    img_out = os.path.join(out, 'image.npy')
    lbl_out = os.path.join(out, 'label.npy')

    # Resume support
    if resume and os.path.exists(img_out) and os.path.exists(lbl_out):
        return {'pid': pid, 'skipped': True}

    data_item = {
        'image_0': os.path.join(pdir, 'image_acq0.nii.gz'),  # P1 early
        'image_1': os.path.join(pdir, 'image_acq1.nii.gz'),  # P2 mid
        'image_2': os.path.join(pdir, 'image_acq2.nii.gz'),  # P3 late
        'label'  : os.path.join(pdir, 'mask.nii.gz'),
    }

    transforms = Compose([
        LoadImaged(
            keys=['image_0', 'image_1', 'image_2', 'label'],
            image_only=True,
            ensure_channel_first=True,
        ),
        # Resample to 1.5mm isotropic to MATCH old preprocessed data
        # New data is already 1.0mm — we downsample to 1.5mm
        Spacingd(
            keys=['image_0', 'image_1', 'image_2', 'label'],
            pixdim=TARGET_SPACING,
            mode=('bilinear', 'bilinear', 'bilinear', 'nearest'),
        ),
        # Intensity normalization — identical to old pipeline
        ScaleIntensityRangePercentilesd(
            keys=['image_0', 'image_1', 'image_2'],
            lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(
            keys=['image_0', 'image_1', 'image_2'],
            nonzero=True,
            channel_wise=True,
        ),
        # Crop to breast region using image intensity
        # (no breast mask available — use image_1 as foreground source)
        CropForegroundd(
            keys=['image_0', 'image_1', 'image_2', 'label'],
            source_key='image_1',
            margin=5,
        ),
        # Merge 3 channels into single image tensor
        ConcatItemsd(
            keys=['image_0', 'image_1', 'image_2'],
            name='image',
            dim=0,
        ),
        EnsureTyped(
            keys=['image', 'label'],
            dtype=torch.float32,
        ),
    ])

    try:
        result = transforms(data_item)
    except Exception as e:
        print(f"  [ERROR] {pid}: {e}")
        return None

    image = result['image'].numpy()   # (3, D, H, W)
    label = result['label'].numpy()   # (1, D, H, W)

    # Verify tumor survived
    tumor_vox = int((label > 0).sum())
    if tumor_vox == 0:
        print(f"  [WARN] {pid}: 0 tumor voxels after preprocessing — skipping")
        return None

    np.save(img_out, image.astype(np.float32))
    np.save(lbl_out, label.astype(np.float32))

    return {
        'pid'       : pid,
        'shape'     : image.shape,
        'tumor_vox' : tumor_vox,
        'tumor_ratio': float(tumor_vox / label.size * 100),
        'skipped'   : False,
    }


def main():
    print("=" * 60)
    print("  Preprocessing ISPY-1 new dataset")
    print(f"  Target spacing : {TARGET_SPACING} mm")
    print(f"  Output root    : {OUTPUT_ROOT}")
    print("=" * 60)

    # Find complete patients
    patients, skipped_list = find_complete_patients(NEW_DATA_ROOT)
    if skipped_list:
        print(f"\nSkipped patients (missing files):")
        for pid, missing in skipped_list[:5]:
            print(f"  {pid}: missing {missing}")

    # Split into train/val/test
    train_pts, val_pts, test_pts = split_patients(
        patients, TRAIN_RATIO, VAL_RATIO)

    print(f"\nSplit: train={len(train_pts)}, val={len(val_pts)}, test={len(test_pts)}")

    # Process each split
    all_stats = {}
    for split_name, split_pts in [('train', train_pts),
                                   ('val',   val_pts),
                                   ('test',  test_pts)]:
        out_dir = os.path.join(OUTPUT_ROOT, split_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing {split_name}: {len(split_pts)} patients")
        stats = []

        for pt in tqdm(split_pts, desc=split_name):
            result = preprocess_patient(pt, out_dir)
            if result:
                stats.append(result)

        processed = [s for s in stats if not s.get('skipped')]
        all_stats[split_name] = processed
        print(f"  Done: {len(processed)} saved")

    # Summary
    print("\n" + "=" * 60)
    print("  Preprocessing complete")
    print("=" * 60)
    total_processed = sum(len(v) for v in all_stats.values())
    print(f"  Total processed : {total_processed}")

    for split, stats in all_stats.items():
        if stats:
            tv = [s['tumor_vox'] for s in stats]
            tr = [s['tumor_ratio'] for s in stats]
            shapes = [s['shape'] for s in stats]
            Ds = [s[1] for s in shapes]
            print(f"\n  {split.upper()} ({len(stats)} patients):")
            print(f"    Tumor voxels : {min(tv):,} – {max(tv):,}  mean {np.mean(tv):.0f}")
            print(f"    Tumor ratio  : {min(tr):.4f}% – {max(tr):.4f}%  mean {np.mean(tr):.4f}%")
            print(f"    D range      : {min(Ds)} – {max(Ds)} voxels")

    print("\n  Next steps:")
    print("  1. Run generate_dataset_json.py (it scans all of patients_preprocessed/)")
    print("  2. The new patients are automatically included in train/val/test")
    print("  3. Retrain with combined dataset")


if __name__ == "__main__":
    main()