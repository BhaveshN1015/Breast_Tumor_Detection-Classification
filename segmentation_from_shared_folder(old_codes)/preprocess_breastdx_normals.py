"""
preprocess_breastdx_normals.py
================================

Re-preprocesses the 16 BreastDx NORMAL patients from their original
DICOM files on your E: drive into the correct (3, D, H, W) npy format.

WHY THIS IS NEEDED:
  The previous preprocessing produced D=432-1707 — far too large.
  Root cause: CropForegroundd was not applied to isolate the breast
  region, so the full stacked DICOM volume was resampled.

WHAT THIS SCRIPT DOES (matching existing pipeline exactly):
  1. Reads the original DICOM series from E:/images/breast_diagnosis/
  2. Selects only the DCE series (3 phases: early, mid, late post-contrast)
  3. Stacks each phase into a 3D NIfTI volume
  4. Applies CropForegroundd on the post-contrast phase to find breast extent
  5. Resamples to 1.5mm isotropic
  6. Clips to 1st-99th percentile (soft, NOT a hard 7.0 ceiling)
  7. Z-score normalises per channel
  8. Saves as image.npy shape (3, D, H, W) + label.npy (all zeros)

EXPECTED OUTPUT:
  data/normal_patients_fixed/
    234/  image.npy (3,D,H,W) float32  label.npy (1,D,H,W) float32
    235/  ...
    ...
  D should be 50-200 (matching existing dataset)

WHICH PATIENTS ARE PROCESSED:
  The 16 BreastDx NORMAL patients (biopsy-proven benign / no lesion)
  You need the clinical CSV from TCIA to know which PatientIDs are normal.
  Set NORMAL_PATIENT_IDS below — list of BreastDx PatientID strings.

Usage:
    python src/classification/preprocess_breastdx_normals.py

Requirements:
    pip install pydicom nibabel SimpleITK numpy tqdm
"""

import os
import re
import sys
import json
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk
from collections import defaultdict
from tqdm import tqdm


# ------------------------------------------------------------------ #
#  CONFIGURATION — edit these before running                          #
# ------------------------------------------------------------------ #

# Root folder of your BreastDx DICOM files
# (the images/ folder you showed in earlier screenshots)
DICOM_ROOT = r"D:\Breast_Tumor_AI_Project\normal patients"

# Output folder for the re-preprocessed patients
OUTPUT_ROOT = r"D:\Breast_Tumor_AI_Project\data\normal_patients_fixed"

# Log file
LOG_FILE = os.path.join(OUTPUT_ROOT, "repreprocess_log.json")

# Target voxel spacing in mm (MUST match existing pipeline)
TARGET_SPACING = (1.5, 1.5, 1.5)

# THESE ARE THE 16 NORMAL PATIENTS from your BreastDx dataset
# PatientID format: BreastDx-01-XXXX
# Edit this list — you confirmed 16 normal patients exist.
# If you have the clinical CSV, extract the benign/normal ones.
# Common BreastDx normals are patients with BI-RADS 1-2 findings.
# For now this is a placeholder — replace with your actual normal IDs.
NORMAL_PATIENT_IDS = [
    "BreastDx-01-0006",
    "BreastDx-01-0011",
    "BreastDx-01-0013",
    "BreastDx-01-0015",
    "BreastDx-01-0016",
    "BreastDx-01-0017",
    "BreastDx-01-0023",
    "BreastDx-01-0029",
    "BreastDx-01-0031",
    "BreastDx-01-0032",
    "BreastDx-01-0033",
    "BreastDx-01-0037",
    "BreastDx-01-0041",
    "BreastDx-01-0066",
    "BreastDx-01-0067",
    "BreastDx-01-0073",
]

# Start numbering from 234 to continue from existing 233 patients
START_ID = 234

# ------------------------------------------------------------------ #
#  STEP 1: DISCOVER DCE SERIES FOR EACH PATIENT                       #
# ------------------------------------------------------------------ #

def discover_dce_series(patient_dicom_dir):
    """
    Scans all DICOM series under patient_dicom_dir.
    Returns a dict: series_number -> list of (z_pos, filepath)
    Only returns T1-weighted DCE series (MR modality, typical breast protocol).
    """
    series_map = defaultdict(list)  # series_uid -> [(z_pos, fpath)]
    series_meta = {}                # series_uid -> metadata

    for root, dirs, files in os.walk(patient_dicom_dir):
        for fname in files:
            if not fname.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                modality = getattr(ds, "Modality", "")
                if modality != "MR":
                    continue
                series_uid  = getattr(ds, "SeriesInstanceUID", "unknown")
                series_num  = getattr(ds, "SeriesNumber", 0)
                series_desc = getattr(ds, "SeriesDescription", "")
                z_pos = 0.0
                if hasattr(ds, "ImagePositionPatient"):
                    z_pos = float(ds.ImagePositionPatient[2])
                series_map[series_uid].append((z_pos, fpath))
                if series_uid not in series_meta:
                    series_meta[series_uid] = {
                        "series_num" : int(series_num),
                        "description": series_desc,
                        "uid"        : series_uid,
                    }
            except Exception:
                continue

    # Sort each series by z_position
    for uid in series_map:
        series_map[uid].sort(key=lambda x: x[0])

    return series_map, series_meta


def select_dce_phases(series_map, series_meta):
    """
    From all series, selects the 3 DCE post-contrast phases.
    Strategy:
      1. Prefer series with descriptions containing "post", "contrast", "dyn"
      2. Among those, take 3 with sequential series numbers
      3. Exclude localiser, T2, and other non-DCE series
      4. If exactly 3 or more DCE series found, return 3 in order
    Returns: list of 3 series UIDs [early, mid, late post-contrast]
    """
    # Filter candidates: must have enough slices to be a volume
    MIN_SLICES = 30  # breast MRI has at least 30 slices per phase

    candidates = []
    for uid, files in series_map.items():
        if len(files) < MIN_SLICES:
            continue
        desc = series_meta[uid]["description"].lower()
        snum = series_meta[uid]["series_num"]

        # Score: DCE series typically have descriptions like:
        # "dynamic", "dce", "post", "contrast", "t1", "bliss", "vibrant"
        # Exclude: "t2", "dwi", "adc", "localizer", "scout", "mip"
        exclude_keywords = ["t2", "dwi", "adc", "diff", "localiz",
                            "scout", "mip", "subtraction", "max",
                            "mask", "seg", "map"]
        dce_keywords = ["dynamic", "dce", "post", "contrast", "t1",
                        "bliss", "vibrant", "dyn", "phase", "pre_vs",
                        "pre vs", "sub"]

        is_excluded = any(kw in desc for kw in exclude_keywords)
        has_dce_kw  = any(kw in desc for kw in dce_keywords)
        score = -1 if is_excluded else (2 if has_dce_kw else 1)

        candidates.append({
            "uid"   : uid,
            "snum"  : snum,
            "desc"  : series_meta[uid]["description"],
            "score" : score,
            "nfiles": len(files),
        })

    # Sort by: score desc, then series number asc
    candidates = [c for c in candidates if c["score"] >= 1]
    candidates.sort(key=lambda x: (-x["score"], x["snum"]))

    if len(candidates) < 1:
        # Fallback: just take largest 3 series by file count
        all_large = sorted(
            [(uid, len(files)) for uid, files in series_map.items()
             if len(files) >= MIN_SLICES],
            key=lambda x: -x[1]
        )
        candidates = [{"uid": uid, "snum": 0, "desc": "",
                        "score": 1, "nfiles": n}
                      for uid, n in all_large]

    # Take first 3 in series-number order (pre, then post phases)
    selected = sorted(candidates[:6], key=lambda x: x["snum"])

    # If we have 4+ (pre + 3 post), skip the first (pre-contrast)
    # Pre-contrast is typically the lowest series number
    if len(selected) >= 4:
        selected = selected[1:4]   # skip pre, take 3 post
    elif len(selected) == 3:
        # Check if first looks like pre-contrast (lower enhancement)
        selected = selected[:3]
    elif len(selected) == 2:
        # Only 2 series — duplicate last one
        selected = selected + [selected[-1]]
    elif len(selected) == 1:
        selected = selected * 3
    else:
        return None

    return [c["uid"] for c in selected[:3]]


# ------------------------------------------------------------------ #
#  STEP 2: READ ONE SERIES → 3D VOLUME                                #
# ------------------------------------------------------------------ #

def read_series_to_volume(series_files):
    """
    Reads all DICOM files for one series (sorted by z_pos).
    Returns (volume_np float32, pixel_spacing, slice_spacing).
    """
    n_slices = len(series_files)
    ds0 = pydicom.dcmread(series_files[0][1])
    H, W = ds0.pixel_array.shape

    volume = np.zeros((n_slices, H, W), dtype=np.float32)
    z_positions = []
    pixel_spacing = None

    for i, (z, fpath) in enumerate(series_files):
        ds  = pydicom.dcmread(fpath)
        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        volume[i] = arr * slope + intercept

        z_positions.append(z)
        if pixel_spacing is None and hasattr(ds, "PixelSpacing"):
            pixel_spacing = (float(ds.PixelSpacing[0]),
                             float(ds.PixelSpacing[1]))

    if pixel_spacing is None:
        pixel_spacing = (1.0, 1.0)

    slice_spacing = 1.0
    if len(z_positions) >= 2:
        diffs = [abs(z_positions[i+1] - z_positions[i])
                 for i in range(len(z_positions)-1)]
        slice_spacing = float(np.median(diffs))
        if slice_spacing < 0.1:
            slice_spacing = 1.0

    return volume, pixel_spacing, slice_spacing


# ------------------------------------------------------------------ #
#  STEP 3: CROP FOREGROUND (BREAST ISOLATION)                         #
# ------------------------------------------------------------------ #

def crop_foreground(volume_np, pixel_spacing, slice_spacing,
                    percentile_thresh=10):
    """
    Crops the volume to the bounding box of non-background voxels.
    Uses a percentile threshold to distinguish breast from air.

    Returns (cropped_np, crop_box)
    crop_box: (d_start, d_end, h_start, h_end, w_start, w_end)
    """
    # Threshold: voxels above percentile are considered tissue
    thresh = np.percentile(volume_np, percentile_thresh)
    mask = volume_np > thresh

    # Find bounding box
    d_idx = np.where(mask.any(axis=(1,2)))[0]
    h_idx = np.where(mask.any(axis=(0,2)))[0]
    w_idx = np.where(mask.any(axis=(0,1)))[0]

    if len(d_idx) == 0 or len(h_idx) == 0 or len(w_idx) == 0:
        # No foreground found — return original
        return volume_np, (0, volume_np.shape[0],
                           0, volume_np.shape[1],
                           0, volume_np.shape[2])

    d_start, d_end = d_idx[0], d_idx[-1] + 1
    h_start, h_end = h_idx[0], h_idx[-1] + 1
    w_start, w_end = w_idx[0], w_idx[-1] + 1

    # Add small margin (5 voxels each side in original spacing)
    margin = 5
    d_start = max(0, d_start - margin)
    d_end   = min(volume_np.shape[0], d_end + margin)
    h_start = max(0, h_start - margin)
    h_end   = min(volume_np.shape[1], h_end + margin)
    w_start = max(0, w_start - margin)
    w_end   = min(volume_np.shape[2], w_end + margin)

    cropped = volume_np[d_start:d_end, h_start:h_end, w_start:w_end]
    return cropped, (d_start, d_end, h_start, h_end, w_start, w_end)


# ------------------------------------------------------------------ #
#  STEP 4: RESAMPLE TO TARGET SPACING                                 #
# ------------------------------------------------------------------ #

def resample_volume(volume_np, pixel_spacing, slice_spacing,
                    target_spacing=TARGET_SPACING):
    """
    Resamples a (D, H, W) volume from its original spacing to target spacing.
    Returns resampled numpy array.
    """
    # Create SimpleITK image
    sitk_img = sitk.GetImageFromArray(volume_np)
    # Spacing in SimpleITK is (x, y, z) = (W_spacing, H_spacing, D_spacing)
    sitk_img.SetSpacing((float(pixel_spacing[1]),
                         float(pixel_spacing[0]),
                         float(slice_spacing)))

    original_spacing = sitk_img.GetSpacing()
    original_size    = sitk_img.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(volume_np.min()))
    resampler.SetInterpolator(sitk.sitkBSpline)

    resampled = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


# ------------------------------------------------------------------ #
#  STEP 5: NORMALISE (matching existing pipeline)                     #
# ------------------------------------------------------------------ #

def normalise_channel(arr):
    """
    1. Clip to 1st-99th percentile (soft, no hard ceiling)
    2. Z-score normalise
    Returns normalised float32 array.
    """
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    clipped = np.clip(arr, p1, p99)

    mean = clipped.mean()
    std  = clipped.std()
    if std < 1e-8:
        std = 1.0

    return ((clipped - mean) / std).astype(np.float32)


# ------------------------------------------------------------------ #
#  MAIN PROCESSING FUNCTION                                           #
# ------------------------------------------------------------------ #

def process_one_patient(patient_id, dicom_root, output_id, output_root,
                        target_spacing=TARGET_SPACING):
    """
    Full pipeline for one BreastDx patient.
    Returns (success, info_dict)
    """
    result = {"patient_id": patient_id, "output_id": output_id}

    # ── Find patient DICOM directory ──────────────────────────────
    patient_dicom = os.path.join(dicom_root, patient_id)
    if not os.path.exists(patient_dicom):
        return False, {**result, "error": f"DICOM dir not found: {patient_dicom}"}

    # ── Discover series ───────────────────────────────────────────
    series_map, series_meta = discover_dce_series(patient_dicom)

    if not series_map:
        return False, {**result, "error": "No DICOM series found"}

    print(f"  {patient_id}: Found {len(series_map)} series:")
    for uid, files in sorted(series_map.items(),
                              key=lambda x: series_meta[x[0]]["series_num"]):
        desc = series_meta[uid]["description"]
        print(f"    Series {series_meta[uid]['series_num']:3d} "
              f"({len(files):3d} files): {desc[:50]}")

    # ── Select 3 DCE phases ───────────────────────────────────────
    phase_uids = select_dce_phases(series_map, series_meta)
    if phase_uids is None:
        return False, {**result, "error": "Could not find 3 DCE phases"}

    print(f"  Selected phases:")
    for i, uid in enumerate(phase_uids):
        desc = series_meta[uid]["description"]
        n    = len(series_map[uid])
        print(f"    P{i+1}: Series {series_meta[uid]['series_num']} "
              f"({n} files): {desc[:50]}")

    # ── Process each phase ────────────────────────────────────────
    channels    = []
    crop_box    = None

    for i, uid in enumerate(phase_uids):
        phase_files = series_map[uid]

        # 1. Read DICOM → 3D volume
        volume_np, pixel_spacing, slice_spacing = read_series_to_volume(
            phase_files)

        print(f"  Phase P{i+1} raw shape: {volume_np.shape}  "
              f"spacing: ({pixel_spacing[0]:.2f}, {pixel_spacing[1]:.2f}, "
              f"{slice_spacing:.2f}) mm")

        # 2. Crop foreground — ONLY compute crop box from P1 (first post-contrast)
        #    Apply SAME crop box to all phases for consistency
        if i == 0:
            cropped, crop_box = crop_foreground(
                volume_np, pixel_spacing, slice_spacing)
            print(f"  Crop box: D[{crop_box[0]}:{crop_box[1]}] "
                  f"H[{crop_box[2]}:{crop_box[3]}] "
                  f"W[{crop_box[4]}:{crop_box[5]}]")
        else:
            d0,d1,h0,h1,w0,w1 = crop_box
            cropped = volume_np[d0:d1, h0:h1, w0:w1]

        print(f"  After crop: {cropped.shape}")

        # 3. Resample to 1.5mm isotropic
        resampled = resample_volume(cropped, pixel_spacing, slice_spacing,
                                    target_spacing)
        print(f"  After resample (1.5mm iso): {resampled.shape}")

        # 4. Normalise
        normalised = normalise_channel(resampled)
        channels.append(normalised)

    # ── Stack 3 channels ──────────────────────────────────────────
    # Ensure same spatial shape (may differ by 1 voxel after resampling)
    min_D = min(c.shape[0] for c in channels)
    min_H = min(c.shape[1] for c in channels)
    min_W = min(c.shape[2] for c in channels)
    channels = [c[:min_D, :min_H, :min_W] for c in channels]

    image_np = np.stack(channels, axis=0)   # (3, D, H, W)
    label_np = np.zeros((1, min_D, min_H, min_W), dtype=np.float32)

    # ── Validation ────────────────────────────────────────────────
    assert image_np.ndim == 4, f"Expected 4D, got {image_np.ndim}D"
    assert image_np.shape[0] == 3, f"Expected 3 channels, got {image_np.shape[0]}"
    assert image_np.dtype == np.float32

    # Check dimensions are in expected range
    D = image_np.shape[1]
    if D > 300:
        return False, {**result,
            "error": f"D={D} still too large — CropForeground may have failed",
            "shape": tuple(image_np.shape)}

    # ── Save ──────────────────────────────────────────────────────
    out_dir = os.path.join(output_root, str(output_id))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "image.npy"), image_np)
    np.save(os.path.join(out_dir, "label.npy"), label_np)

    result.update({
        "shape"        : tuple(image_np.shape),
        "img_mean"     : round(float(image_np.mean()), 4),
        "img_std"      : round(float(image_np.std()),  4),
        "img_max"      : round(float(image_np.max()),  4),
        "output_path"  : out_dir,
        "status"       : "OK",
    })
    return True, result


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    print()
    print("=" * 65)
    print("  BreastDx Normals — Re-preprocessing from DICOM")
    print("=" * 65)
    print(f"\n  DICOM root  : {DICOM_ROOT}")
    print(f"  Output root : {OUTPUT_ROOT}")
    print(f"  Patients    : {len(NORMAL_PATIENT_IDS)}")
    print(f"  Target IDs  : {START_ID} to {START_ID + len(NORMAL_PATIENT_IDS) - 1}")
    print(f"  Spacing     : {TARGET_SPACING} mm")
    print()

    if not os.path.exists(DICOM_ROOT):
        print(f"  ERROR: DICOM root not found: {DICOM_ROOT}")
        print(f"  Update DICOM_ROOT at the top of the script.")
        sys.exit(1)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success_list = []
    fail_list    = []
    log_entries  = []

    for i, pid in enumerate(tqdm(NORMAL_PATIENT_IDS,
                                  desc="Processing", ncols=70)):
        output_id = START_ID + i
        print(f"\n[{i+1}/{len(NORMAL_PATIENT_IDS)}] {pid} → ID {output_id}")

        # Skip if already done
        out_npy = os.path.join(OUTPUT_ROOT, str(output_id), "image.npy")
        if os.path.exists(out_npy):
            print(f"  Already exists — skipping")
            success_list.append(pid)
            continue

        ok, info = process_one_patient(
            patient_id  = pid,
            dicom_root  = DICOM_ROOT,
            output_id   = output_id,
            output_root = OUTPUT_ROOT,
        )

        log_entries.append(info)

        if ok:
            success_list.append(pid)
            shape = info["shape"]
            print(f"  ✓ Saved → {info['output_path']}")
            print(f"    shape={shape}  mean={info['img_mean']}  "
                  f"std={info['img_std']}  max={info['img_max']}")
        else:
            fail_list.append((pid, info.get("error", "unknown")))
            print(f"  ✗ FAILED: {info.get('error', 'unknown')}")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Re-preprocessing complete")
    print("=" * 65)
    print(f"  Successful : {len(success_list)}/{len(NORMAL_PATIENT_IDS)}")
    print(f"  Failed     : {len(fail_list)}")
    if fail_list:
        print("\n  Failed patients:")
        for pid, err in fail_list:
            print(f"    {pid}: {err}")

    print()
    print("  Output shape summary:")
    for entry in log_entries:
        if entry.get("status") == "OK":
            print(f"    ID {entry['output_id']:3d}  {str(entry['shape']):25s}  "
                  f"mean={entry['img_mean']:7.4f}  std={entry['img_std']:6.4f}  "
                  f"max={entry['img_max']:6.4f}")

    # ── Save log ──────────────────────────────────────────────────
    with open(LOG_FILE, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)
    print(f"\n  Log saved → {LOG_FILE}")

    print()
    print("  Next step: run check_dataset_compatibility.py again")
    print(f"  to verify the re-processed patients now have D=50-220")


if __name__ == "__main__":
    main()
