import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
PATIENTS_ROOT = "data/patients"
OUTPUT_ROOT   = "data/Segmentation_dataset"

SPLITS = {
    "train" : os.path.join(PATIENTS_ROOT, "train"),
    "val"   : os.path.join(PATIENTS_ROOT, "val"),
    "test"  : os.path.join(PATIENTS_ROOT, "test"),
}

OUTPUT = {
    "train" : {
        "images" : os.path.join(OUTPUT_ROOT, "train/images"),
        "masks"  : os.path.join(OUTPUT_ROOT, "train/masks"),
    },
    "val" : {
        "images" : os.path.join(OUTPUT_ROOT, "val/images"),
        "masks"  : os.path.join(OUTPUT_ROOT, "val/masks"),
    },
    "test" : {
        "images" : os.path.join(OUTPUT_ROOT, "test/images"),
        "masks"  : os.path.join(OUTPUT_ROOT, "test/masks"),
    },
}

for split in OUTPUT:
    for folder in OUTPUT[split].values():
        os.makedirs(folder, exist_ok=True)

# -----------------------------
# PARAMETERS
# -----------------------------
IMAGE_SIZE = 256

# 3-CHANNEL: P1=early post-contrast, P2=mid, P3=late
# Stacked as channels → model sees contrast enhancement over time
# This is how radiologists read DCE-MRI — temporal pattern is key
MRI_SEQUENCES = ["P1", "P2", "P3"]   # order matters: ch0=P1, ch1=P2, ch2=P3

print("=" * 60)
print("  Dataset Preparation — 3-Channel DCE-MRI")
print("=" * 60)
print(f"  Sequences (as channels) : {MRI_SEQUENCES}")
print(f"  Image size              : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Output format           : 3-channel PNG (P1+P2+P3 stacked)")
print(f"  CLAHE                   : ON — per channel")
print(f"  Tumor filter            : NONE — all tumor sizes kept")
print("=" * 60)
print()


def apply_clahe(image_uint8):
    """Adaptive contrast enhancement — makes tumor visible across sequences."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_uint8)


def find_nii(path_without_ext):
    for ext in [".nii.gz", ".nii"]:
        full = path_without_ext + ext
        if os.path.exists(full):
            return full
    return None


def normalize_slice(slice_2d):
    """Normalize a 2D array to uint8 (0-255)."""
    if slice_2d.max() > slice_2d.min():
        s = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    else:
        s = np.zeros_like(slice_2d)
    return (s * 255).astype(np.uint8)


def process_patient(patient_id, patient_path, split):
    """
    For each slice index, load P1+P2+P3 simultaneously and
    save as a single 3-channel PNG (BGR order: ch0=P1, ch1=P2, ch2=P3).
    One image file per slice instead of three separate files.
    """
    # Load GT mask
    mask_path = find_nii(os.path.join(patient_path, "GT"))
    if mask_path is None:
        print(f"  [SKIP] GT not found: patient {patient_id}")
        return 0

    mask_volume = nib.load(mask_path).get_fdata()

    # Load all 3 sequence volumes first
    volumes = {}
    for seq in MRI_SEQUENCES:
        mri_path = find_nii(os.path.join(patient_path, seq))
        if mri_path is None:
            print(f"  [SKIP] {seq} not found: patient {patient_id}")
            return 0
        volumes[seq] = nib.load(mri_path).get_fdata()

    # All volumes must have the same depth
    depths = [volumes[seq].shape[2] for seq in MRI_SEQUENCES]
    if len(set(depths)) != 1:
        print(f"  [SKIP] Depth mismatch for patient {patient_id}: {depths}")
        return 0

    depth = depths[0]
    saved = 0

    for i in range(depth):

        # --- Build 3-channel image ---
        channels = []
        for seq in MRI_SEQUENCES:
            sl = volumes[seq][:, :, i]
            sl = normalize_slice(sl)
            sl = cv2.resize(sl, (IMAGE_SIZE, IMAGE_SIZE))
            sl = apply_clahe(sl)      # CLAHE per channel
            channels.append(sl)

        # Stack as 3-channel: shape (256, 256, 3)
        # cv2.imwrite saves in BGR order — we use P1=B, P2=G, P3=R
        # When loading back with cv2.IMREAD_COLOR it returns (H,W,3) BGR
        three_channel = cv2.merge(channels)   # (256, 256, 3)

        # --- Process mask ---
        mask_slice = mask_volume[:, :, i]
        mask_slice = cv2.resize(
            mask_slice, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        mask_slice = (mask_slice > 0).astype(np.uint8) * 255

        # --- Save ---
        # Naming: patient{id}_slice{iii}.png  (no sequence in name — it IS all 3)
        img_name  = f"patient{patient_id}_slice{i:03d}.png"
        mask_name = f"patient{patient_id}_slice{i:03d}_mask.png"

        cv2.imwrite(os.path.join(OUTPUT[split]["images"], img_name), three_channel)
        cv2.imwrite(os.path.join(OUTPUT[split]["masks"],  mask_name), mask_slice)

        saved += 1

    return saved


# -----------------------------
# PROCESS ALL SPLITS
# -----------------------------
grand_total = 0

for split, split_path in SPLITS.items():

    if not os.path.exists(split_path):
        print(f"[WARNING] Folder not found: {split_path}")
        continue

    patients = sorted(
        [p for p in os.listdir(split_path)
         if os.path.isdir(os.path.join(split_path, p))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    print(f"Processing {split.upper()} — {len(patients)} patients")

    split_total = 0

    for patient_id in tqdm(patients, desc=split):
        patient_path = os.path.join(split_path, patient_id)
        saved = process_patient(patient_id, patient_path, split)
        split_total += saved

    grand_total += split_total

    n_img  = len(os.listdir(OUTPUT[split]["images"]))
    n_mask = len(os.listdir(OUTPUT[split]["masks"]))

    print(f"  Slices saved : {split_total}")
    print(f"  Images       : {n_img}")
    print(f"  Masks        : {n_mask}")
    print(f"  Aligned      : {'YES' if n_img == n_mask else 'NO — CHECK THIS'}")
    print()

print("=" * 60)
print("  Preparation Complete")
print("=" * 60)
print(f"  Total slices  : {grand_total}")
print(f"  Train         : {len(os.listdir(OUTPUT['train']['images']))}")
print(f"  Val           : {len(os.listdir(OUTPUT['val']['images']))}")
print(f"  Test          : {len(os.listdir(OUTPUT['test']['images']))}")
print()
print("  Each image is a 3-channel PNG: P1+P2+P3 stacked")
print("  Next step: python src/segmentation/train_unet.py")