"""
prepare_breastdx_npy.py
=======================

Converts BreastDx NIfTI files (BreastDx-01-XXXX.nii or .nii.gz) into the
exact .npy format used by the rest of this project.

Output format (matches data/patients_combined exactly):
  image.npy  →  shape (3, D, H, W)  float32  z-score normalised
  label.npy  →  shape (1, D, H, W)  float32  binary {0.0, 1.0}

Pipeline per patient:
  1. Load NIfTI volume + label mask
  2. Resample both to 1.5 mm isotropic spacing (matches all other patients)
  3. Crop to breast region using label bounding-box + 20-voxel margin
  4. Z-score normalise image (per-channel, non-zero voxels only)
  5. Verify: float32, 3 channels, binary label, tumor present, max < 10
  6. Save image.npy + label.npy to output folder

BreastDx dataset structure assumption:
  Each case has ONE image volume (single phase or multi-phase).
  If your BreastDx files are single-phase (one file per patient),
  the script duplicates the channel to produce a (3, D, H, W) array
  so the model sees the correct input shape.
  If your files ARE already multi-phase (3 separate files per patient),
  set MODE = "multi" below and point PHASE_SUFFIXES to your naming pattern.

Usage:
    1.  Set INPUT_DIR  → folder containing your .nii files (e.g. F:/converted_nii)
    2.  Set LABEL_DIR  → folder containing matching label .nii files
                         (set to None if labels will be added later — dummy zeros used)
    3.  Set OUTPUT_ROOT → where to write patient folders
                          (e.g. data/patients_combined — uses next available IDs)
    4.  Set START_ID   → first numeric patient ID to assign (e.g. 234 if you have 233)
    5.  Run:  python prepare_breastdx_npy.py

Dependencies (already in your ml env):
    nibabel, numpy, scipy, tqdm
    pip install nibabel --break-system-packages  (if missing)
"""

import os
import re
import glob
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes

# ------------------------------------------------------------------ #
#  >>>  CONFIGURE THESE BEFORE RUNNING  <<<                           #
# ------------------------------------------------------------------ #

# Folder on your USB drive containing the .nii files
INPUT_DIR   = r"F:\converted_nii"

# Folder containing matching label/mask .nii files.
# Naming convention: if image is BreastDx-01-0006.nii,
# label should be BreastDx-01-0006_label.nii  (or _seg, _mask — set below)
# Set to None if you have no labels yet → dummy zero masks will be written
LABEL_DIR   = None          # e.g. r"F:\converted_nii_labels"

# Suffix that distinguishes label files from image files (ignored if LABEL_DIR is None)
LABEL_SUFFIX = "_label"     # e.g. "_seg", "_mask", "_gt"

# Where to save the converted patients
# Set to your existing combined folder so they integrate automatically
OUTPUT_ROOT = "data/patients_combined"

# Which split to assign new patients to (can redistribute later)
# "train" | "val" | "test"
DEFAULT_SPLIT = "train"

# First numeric patient ID to assign.
# Set this to max_existing_id + 1 (you have 233, so use 234)
START_ID    = 234

# Target voxel spacing after resampling (must match existing data)
TARGET_SPACING_MM = (1.5, 1.5, 1.5)

# Bounding-box crop margin in voxels (added on each side around tumor/breast)
CROP_MARGIN = 20

# Channel mode:
#   "single"  → each .nii is ONE phase; script triplicates it → (3,D,H,W)
#                This is safe — the model will learn from DCE dynamics
#                once you have real multi-phase data.
#   "multi"   → set PHASE_SUFFIXES below; script stacks 3 files per patient
MODE = "single"

# Only used when MODE = "multi"
# e.g. if files are BreastDx-01-0006_P1.nii, BreastDx-01-0006_P2.nii, ...
PHASE_SUFFIXES = ["_P1", "_P2", "_P3"]

# Minimum tumor voxel count to accept a patient (after resampling)
# Patients below this are flagged but still saved (you can filter later)
MIN_TUMOR_VOXELS = 50

# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def resample_volume(volume_np, original_spacing, target_spacing, order=3):
    """
    Resample a 3D numpy array from original_spacing to target_spacing.
    order=3 (cubic) for images, order=0 (nearest) for masks.
    original_spacing and target_spacing are (z, y, x) tuples in mm.
    """
    zoom_factors = [
        orig / tgt
        for orig, tgt in zip(original_spacing, target_spacing)
    ]
    resampled = zoom(volume_np, zoom_factors, order=order, prefilter=(order > 1))
    return resampled.astype(np.float32)


def get_spacing_from_nifti(img):
    """
    Extract voxel spacing (z, y, x) in mm from a NIfTI image.
    NIfTI header pixdim is [qfac, x, y, z, t, ...] — we reorder to (z,y,x).
    """
    pixdim = img.header.get_zooms()
    if len(pixdim) >= 3:
        # pixdim gives (x, y, z) spacing — reorder to (z, y, x) for numpy [D,H,W]
        spacing = (float(pixdim[2]), float(pixdim[1]), float(pixdim[0]))
    else:
        print("  [WARN] Could not read spacing — assuming 1mm isotropic")
        spacing = (1.0, 1.0, 1.0)
    return spacing


def crop_to_region(volume, mask, margin=20):
    """
    Crop volume and mask to the bounding box of non-zero mask voxels + margin.
    If mask is all zeros (no label), crops to non-zero image region instead.
    Returns cropped (volume, mask).
    """
    if mask.sum() > 0:
        region = mask > 0
    else:
        region = volume > 0

    coords = np.argwhere(region)
    if len(coords) == 0:
        return volume, mask

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    D, H, W = volume.shape
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(D, z1 + margin)
    y1 = min(H, y1 + margin)
    x1 = min(W, x1 + margin)

    return volume[z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1]


def zscore_normalise(channel_3d):
    """
    Z-score normalise a single (D,H,W) channel using non-zero voxels only.
    Non-zero mask avoids pulling the mean down with background air voxels.
    Clips to [-7, 7] to prevent outlier explosion then scales.
    """
    nonzero = channel_3d[channel_3d != 0]
    if len(nonzero) == 0:
        return channel_3d.astype(np.float32)
    mu  = nonzero.mean()
    std = nonzero.std()
    if std < 1e-6:
        std = 1.0
    normed = (channel_3d - mu) / std
    normed = np.clip(normed, -7.0, 7.0)
    return normed.astype(np.float32)


def verify_patient(image_npy, label_npy, patient_id):
    """
    Run the 8 compatibility checks matching data_3d.py expectations.
    Returns (passed, flags_list).
    """
    flags = []

    if image_npy.dtype != np.float32:
        flags.append(f"DTYPE_IMAGE={image_npy.dtype}")
    if label_npy.dtype != np.float32:
        flags.append(f"DTYPE_LABEL={label_npy.dtype}")
    if image_npy.shape[0] != 3:
        flags.append(f"CHANNELS={image_npy.shape[0]}")
    unique_vals = np.unique(label_npy)
    if not all(v in [0.0, 1.0] for v in unique_vals):
        flags.append(f"LABEL_NOT_BINARY={unique_vals[:5]}")

    tumor_voxels = int((label_npy > 0).sum())
    if tumor_voxels == 0:
        flags.append("NO_TUMOR")
    elif tumor_voxels < MIN_TUMOR_VOXELS:
        flags.append(f"TINY_TUMOR={tumor_voxels}")

    img_max = float(image_npy.max())
    if img_max > 10.0:
        flags.append(f"MAX_TOO_HIGH={img_max:.2f}")

    D, H, W = image_npy.shape[1], image_npy.shape[2], image_npy.shape[3]
    if D < 40:
        flags.append(f"THIN_D={D}")
    if max(H, W) > 352:
        flags.append(f"LARGE_HW={H}x{W}")

    return len(flags) == 0, flags


def find_nii_files(directory):
    """Return sorted list of all .nii and .nii.gz files in directory."""
    nii  = glob.glob(os.path.join(directory, "*.nii"))
    niigz = glob.glob(os.path.join(directory, "*.nii.gz"))
    all_files = sorted(nii + niigz)
    return all_files


def extract_patient_key(filepath):
    """
    Extract a stable key from the filename for matching image to label.
    e.g. 'BreastDx-01-0006.nii' → 'BreastDx-01-0006'
    """
    base = os.path.basename(filepath)
    key  = base.replace(".nii.gz", "").replace(".nii", "")
    return key


# ------------------------------------------------------------------ #
#  MAIN CONVERSION                                                     #
# ------------------------------------------------------------------ #

def main():
    print("=" * 65)
    print("  BreastDx NIfTI → .npy converter")
    print("  Output format: image(3,D,H,W) label(1,D,H,W) float32")
    print("=" * 65)
    print(f"  Input dir    : {INPUT_DIR}")
    print(f"  Label dir    : {LABEL_DIR if LABEL_DIR else 'None (dummy zeros)'}")
    print(f"  Output root  : {OUTPUT_ROOT}")
    print(f"  Split        : {DEFAULT_SPLIT}")
    print(f"  Start ID     : {START_ID}")
    print(f"  Mode         : {MODE}")
    print(f"  Target spacing: {TARGET_SPACING_MM} mm")
    print()

    # ---- Find all image files ----
    if MODE == "single":
        # Filter out label files if they're in the same folder
        all_nii = find_nii_files(INPUT_DIR)
        if LABEL_DIR and INPUT_DIR == LABEL_DIR:
            all_nii = [f for f in all_nii if LABEL_SUFFIX not in f]
        image_files = sorted(all_nii)
    elif MODE == "multi":
        # Find files matching first phase suffix only, derive others
        p1_files = [
            f for f in find_nii_files(INPUT_DIR)
            if PHASE_SUFFIXES[0] in os.path.basename(f)
        ]
        image_files = sorted(p1_files)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    if not image_files:
        print(f"[ERROR] No .nii files found in {INPUT_DIR}")
        print("  Check INPUT_DIR path and ensure files have .nii or .nii.gz extension.")
        return

    print(f"Found {len(image_files)} image files to convert.\n")

    # ---- Build label lookup if LABEL_DIR is set ----
    label_lookup = {}
    if LABEL_DIR and os.path.exists(LABEL_DIR):
        label_files = find_nii_files(LABEL_DIR)
        for lf in label_files:
            key = extract_patient_key(lf).replace(LABEL_SUFFIX, "")
            label_lookup[key] = lf
        print(f"Found {len(label_lookup)} label files in {LABEL_DIR}")

    # ---- Output split folder ----
    split_dir = os.path.join(OUTPUT_ROOT, DEFAULT_SPLIT)
    os.makedirs(split_dir, exist_ok=True)

    # ---- Conversion loop ----
    patient_id = START_ID
    success_count = 0
    flagged_patients = []
    summary_rows = []

    for img_path in tqdm(image_files, desc="Converting", ncols=90):

        img_key = extract_patient_key(img_path)
        tqdm.write(f"\n  [{patient_id}] {img_key}")

        # ── 1. Load image ──────────────────────────────────────────
        try:
            img_nib  = nib.load(img_path)
            img_data = img_nib.get_fdata(dtype=np.float32)   # (H, W, D) or (H,W,D,T)
        except Exception as e:
            tqdm.write(f"  [ERROR] Could not load image: {e} — skipping")
            patient_id += 1
            continue

        # NIfTI convention: data is (x, y, z) = (W, H, D) in numpy
        # We want (D, H, W) — transpose accordingly
        if img_data.ndim == 4:
            # Multi-phase volume already (W, H, D, T) or (T, W, H, D)
            # Try to figure out which axis is time
            if img_data.shape[3] >= 3:
                # (W, H, D, T) → take first 3 time points as channels
                img_data = img_data[..., :3]             # (W, H, D, 3)
                img_data = np.transpose(img_data, (2,1,0,3))  # (D, H, W, 3)
                # Put channels first
                img_data = np.transpose(img_data, (3,0,1,2))  # (3, D, H, W)
                tqdm.write(f"  4D volume detected — using first 3 time points as channels")
                multi_phase = True
            else:
                tqdm.write(f"  [WARN] 4D volume with <3 time points, using first volume")
                img_data = img_data[..., 0]  # (W, H, D)
                multi_phase = False
        else:
            multi_phase = False

        if img_data.ndim == 3:
            # Single 3D volume: (W, H, D) → (D, H, W)
            img_data = np.transpose(img_data, (2, 1, 0))   # (D, H, W)

        # ── 2. Load or create label ────────────────────────────────
        label_data = None
        if img_key in label_lookup:
            try:
                lbl_nib   = nib.load(label_lookup[img_key])
                label_data = lbl_nib.get_fdata(dtype=np.float32)
                if label_data.ndim == 4:
                    label_data = label_data[..., 0]
                label_data = np.transpose(label_data, (2, 1, 0))  # (D, H, W)
                label_data = (label_data > 0.5).astype(np.float32)
                tqdm.write(f"  Label loaded: {label_data.sum():.0f} tumor voxels")
            except Exception as e:
                tqdm.write(f"  [WARN] Could not load label: {e} — using zeros")
                label_data = None
        else:
            if LABEL_DIR:
                tqdm.write(f"  [WARN] No label found for {img_key} — using zeros")

        if label_data is None:
            # Create dummy zero label — same shape as image volume
            if img_data.ndim == 3:
                label_data = np.zeros(img_data.shape, dtype=np.float32)
            else:
                label_data = np.zeros(img_data.shape[1:], dtype=np.float32)

        # ── 3. Get voxel spacing ───────────────────────────────────
        spacing = get_spacing_from_nifti(img_nib)
        tqdm.write(f"  Original spacing : {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm")

        # ── 4. Resample to 1.5mm isotropic ────────────────────────
        if multi_phase and img_data.ndim == 4:
            # img_data is (3, D, H, W) — resample each channel independently
            channels_resampled = []
            for ch in range(3):
                ch_vol = img_data[ch]   # (D, H, W)
                ch_res = resample_volume(ch_vol, spacing, TARGET_SPACING_MM, order=3)
                channels_resampled.append(ch_res)
            img_resampled = np.stack(channels_resampled, axis=0)  # (3, D, H, W)
            # Use channel 0 shape for label resampling
            ref_shape = channels_resampled[0].shape
            lbl_resampled = resample_volume(label_data, spacing, TARGET_SPACING_MM, order=0)
        else:
            # img_data is (D, H, W) — single phase
            img_resampled = resample_volume(img_data, spacing, TARGET_SPACING_MM, order=3)
            lbl_resampled = resample_volume(label_data, spacing, TARGET_SPACING_MM, order=0)

        # Binarise label after resampling (nearest-neighbour can produce 0/1 only,
        # but cubic resampled values may be fractional if someone resampled mask wrong)
        lbl_resampled = (lbl_resampled > 0.5).astype(np.float32)

        tqdm.write(f"  Resampled shape  : {img_resampled.shape if img_resampled.ndim==3 else img_resampled.shape[1:]}")

        # ── 5. Crop to breast region ───────────────────────────────
        if img_resampled.ndim == 3:
            img_crop, lbl_crop = crop_to_region(img_resampled, lbl_resampled, CROP_MARGIN)
        else:
            # Crop each channel with same bounding box derived from label
            lbl_crop_temp = lbl_resampled
            if lbl_resampled.sum() > 0:
                region = lbl_resampled > 0
            else:
                region = img_resampled[0] > 0

            coords = np.argwhere(region)
            if len(coords) > 0:
                D_r, H_r, W_r = img_resampled.shape[1:]
                z0,y0,x0 = coords.min(axis=0)
                z1,y1,x1 = coords.max(axis=0) + 1
                z0 = max(0, z0 - CROP_MARGIN);  z1 = min(D_r, z1 + CROP_MARGIN)
                y0 = max(0, y0 - CROP_MARGIN);  y1 = min(H_r, y1 + CROP_MARGIN)
                x0 = max(0, x0 - CROP_MARGIN);  x1 = min(W_r, x1 + CROP_MARGIN)
                img_crop  = img_resampled[:, z0:z1, y0:y1, x0:x1]
                lbl_crop  = lbl_resampled[z0:z1, y0:y1, x0:x1]
            else:
                img_crop = img_resampled
                lbl_crop = lbl_resampled

        tqdm.write(f"  Cropped shape    : {img_crop.shape if img_crop.ndim==3 else img_crop.shape[1:]}")

        # ── 6. Build 3-channel array ───────────────────────────────
        if img_crop.ndim == 3:
            # Single phase: triplicate to (3, D, H, W)
            # Rationale: the model expects 3 channels (P1/P2/P3 DCE phases)
            # If you only have 1 phase, triplication is the correct fallback
            # so the model doesn't receive an unexpected input shape.
            img_3ch = np.stack([img_crop, img_crop, img_crop], axis=0)  # (3,D,H,W)
            tqdm.write(f"  Single phase → triplicated to (3,D,H,W)")
        else:
            img_3ch = img_crop   # already (3, D, H, W)

        # ── 7. Z-score normalise per channel ──────────────────────
        for ch in range(3):
            img_3ch[ch] = zscore_normalise(img_3ch[ch])

        img_3ch = img_3ch.astype(np.float32)

        # ── 8. Final label shape: (1, D, H, W) ────────────────────
        lbl_final = lbl_crop[np.newaxis, :, :, :].astype(np.float32)  # (1,D,H,W)

        # ── 9. Verify ──────────────────────────────────────────────
        passed, flags = verify_patient(img_3ch, lbl_final, patient_id)

        D_f, H_f, W_f = img_3ch.shape[1], img_3ch.shape[2], img_3ch.shape[3]
        tumor_v = int(lbl_final.sum())
        img_max = float(img_3ch.max())
        ratio   = tumor_v / (D_f * H_f * W_f) * 100

        tqdm.write(f"  Final shape      : image{img_3ch.shape}  label{lbl_final.shape}")
        tqdm.write(f"  Tumor voxels     : {tumor_v}  ({ratio:.3f}%)")
        tqdm.write(f"  Image max        : {img_max:.3f}")
        tqdm.write(f"  Status           : {'PASS' if passed else 'FLAGGED: ' + str(flags)}")

        # ── 10. Save ───────────────────────────────────────────────
        patient_dir = os.path.join(split_dir, str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)

        np.save(os.path.join(patient_dir, "image.npy"), img_3ch)
        np.save(os.path.join(patient_dir, "label.npy"), lbl_final)

        tqdm.write(f"  Saved to         : {patient_dir}/")

        success_count += 1
        if not passed:
            flagged_patients.append((patient_id, img_key, flags))

        summary_rows.append({
            "patient_id"  : patient_id,
            "source_file" : img_key,
            "split"       : DEFAULT_SPLIT,
            "shape_D"     : D_f,
            "shape_H"     : H_f,
            "shape_W"     : W_f,
            "tumor_voxels": tumor_v,
            "tumor_ratio%": round(ratio, 4),
            "img_max"     : round(img_max, 3),
            "flags"       : "|".join(flags) if flags else "OK",
        })

        patient_id += 1

    # ---- Summary ----
    print()
    print("=" * 65)
    print("  Conversion complete")
    print("=" * 65)
    print(f"  Files processed  : {len(image_files)}")
    print(f"  Saved            : {success_count}")
    print(f"  Flagged          : {len(flagged_patients)}")
    print(f"  Patient IDs      : {START_ID} – {patient_id - 1}")
    print(f"  Output folder    : {split_dir}")
    print()

    if flagged_patients:
        print("  Flagged patients (review before training):")
        for pid, name, fl in flagged_patients:
            print(f"    Patient {pid:4d} [{name}]  flags: {fl}")
        print()

    # ---- Save conversion log ----
    log_path = os.path.join(OUTPUT_ROOT, "breastdx_conversion_log.json")
    with open(log_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"  Conversion log   : {log_path}")

    # ---- Reminder about generate_dataset_json ----
    print()
    print("  Next steps:")
    print("    1. Review flagged patients above")
    print("    2. Re-run generate_dataset_json.py to include new patients in training")
    print("       python src/segmentation_3d/generate_dataset_json.py")
    print("    3. Update START_ID in this script if you convert more batches later")
    print("=" * 65)


if __name__ == "__main__":
    main()
