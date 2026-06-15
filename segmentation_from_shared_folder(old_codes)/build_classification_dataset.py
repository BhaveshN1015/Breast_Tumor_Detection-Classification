"""
build_classification_dataset.py
================================
Builds two clean classification datasets from all sources:

  Stage 1 — Tumor Detection  (tumor-present vs no-malignant-tumor)
  Stage 3 — Malignancy       (malignant vs benign)

OUTPUT STRUCTURE
----------------
data/
  classification_stage1/
    train/
      positive/   <- symlinks or copies of image.npy
      negative/
    val/
      positive/
      negative/
    test/
      positive/
      negative/
    dataset_info.json   <- patient list, labels, source, split

  classification_stage3/
    train/
      malignant/
      benign/
    val/
      malignant/
      benign/
    test/
      malignant/
      benign/
    dataset_info.json

SOURCES
-------
  A) data/patients_combined/   — 233 malignant patients (IDs 1–233)
     Already split into train/val/test subfolders.

  B) data/fast_mri_complete/   — 268 fastMRI patients
     Already split into train/val/test / benign|malignant|normal
     benign   (136) → Stage1 negative, Stage3 benign
     malignant ( 58) → Stage1 positive, Stage3 malignant  [count from screenshots]
     normal   ( 42+) → Stage1 negative only

  C) data/breastdx_normal/     — 16 normal patients (IDs 234–249), flat folder
     Needs splitting 70/15/15 → Stage1 negative only

HOW IT WORKS
------------
  Uses shutil.copy2 to copy image.npy (and label.npy) into the new
  classification folders — so originals are never moved or deleted.
  Re-running is safe: existing files are skipped.

USAGE
-----
  python src/classification/build_classification_dataset.py

  Optional flags:
    --dry-run      Print what would be done without copying anything
    --stage1-only  Build only Stage 1 dataset
    --stage3-only  Build only Stage 3 dataset
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
#  PATHS  —  edit these to match your machine
# ──────────────────────────────────────────────────────────
PROJECT_ROOT      = r"D:\Breast_Tumor_AI_Project"

# Source roots
PATIENTS_COMBINED = os.path.join(PROJECT_ROOT, "data", "patients_combined")
FAST_MRI_ROOT     = r"C:\fast_mri_complete"
BREASTDX_NORMAL   = os.path.join(PROJECT_ROOT, "data", "breastdx_normal")

# Output roots
STAGE1_OUT        = os.path.join(PROJECT_ROOT, "data", "classification_stage1")
STAGE3_OUT        = os.path.join(PROJECT_ROOT, "data", "classification_stage3")

# ──────────────────────────────────────────────────────────
#  SPLIT CONFIG
# ──────────────────────────────────────────────────────────
# BreastDx normals (16 flat patients) need splitting from scratch
BREASTDX_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED    = 42

# ──────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────

def find_npy_patients(folder):
    """Return list of dicts {patient_id, image_path, label_path} for all
    patients under `folder` that have image.npy."""
    patients = []
    if not os.path.isdir(folder):
        return patients
    for pid in sorted(os.listdir(folder)):
        img = os.path.join(folder, pid, "image.npy")
        lbl = os.path.join(folder, pid, "label.npy")
        if os.path.isfile(img):
            patients.append({
                "patient_id":  pid,
                "image_path":  img,
                "label_path":  lbl if os.path.isfile(lbl) else None,
            })
    return patients


def copy_patient(entry, dest_class_folder, dry_run=False):
    """Copy image.npy (and label.npy if present) into dest_class_folder/patient_id/."""
    pid      = entry["patient_id"]
    dest_dir = os.path.join(dest_class_folder, pid)
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    img_dst = os.path.join(dest_dir, "image.npy")
    if not os.path.isfile(img_dst) or dry_run:
        if not dry_run:
            shutil.copy2(entry["image_path"], img_dst)

    if entry["label_path"] and os.path.isfile(entry["label_path"]):
        lbl_dst = os.path.join(dest_dir, "label.npy")
        if not os.path.isfile(lbl_dst) or dry_run:
            if not dry_run:
                shutil.copy2(entry["label_path"], lbl_dst)


def scan_fast_mri(fast_mri_root):
    """Scan fast_mri_complete/{train,val,test}/{benign,malignant,normal}/
    Returns dict: {split: {class: [patient_entries]}}"""
    result = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(fast_mri_root, split)
        if not os.path.isdir(split_dir):
            continue
        result[split] = {}
        for cls in ("benign", "malignant", "normal"):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                patients = find_npy_patients(cls_dir)
                result[split][cls] = patients
    return result


def scan_patients_combined(combined_root):
    """Scan patients_combined/{train,val,test}/ — all malignant.
    Returns dict: {split: [patient_entries]}"""
    result = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(combined_root, split)
        patients  = find_npy_patients(split_dir)
        if patients:
            result[split] = patients
    return result


def split_flat_patients(flat_root, ratios, seed=42):
    """Split a flat folder of patients into train/val/test.
    Returns dict: {split: [patient_entries]}"""
    patients = find_npy_patients(flat_root)
    random.seed(seed)
    random.shuffle(patients)

    n       = len(patients)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    # test gets the remainder

    return {
        "train": patients[:n_train],
        "val":   patients[n_train : n_train + n_val],
        "test":  patients[n_train + n_val :],
    }


# ──────────────────────────────────────────────────────────
#  BUILD STAGE 1  (tumor-present vs no-malignant-tumor)
# ──────────────────────────────────────────────────────────

def build_stage1(combined, fast_mri, breastdx_splits, dry_run=False):
    print("\n" + "="*60)
    print("  BUILDING STAGE 1 — Tumor Detection")
    print("  Positive: malignant patients")
    print("  Negative: benign + normal patients")
    print("="*60)

    info = {"stage": 1, "classes": {"positive": [], "negative": []}}
    counts = {"train": {"positive": 0, "negative": 0},
              "val":   {"positive": 0, "negative": 0},
              "test":  {"positive": 0, "negative": 0}}

    for split in ("train", "val", "test"):
        pos_dir = os.path.join(STAGE1_OUT, split, "positive")
        neg_dir = os.path.join(STAGE1_OUT, split, "negative")
        if not dry_run:
            os.makedirs(pos_dir, exist_ok=True)
            os.makedirs(neg_dir, exist_ok=True)

        # ── POSITIVE: patients_combined (all malignant) ──
        for entry in combined.get(split, []):
            copy_patient(entry, pos_dir, dry_run)
            info["classes"]["positive"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "patients_combined"
            })
            counts[split]["positive"] += 1

        # ── POSITIVE: fastMRI malignant ──
        for entry in fast_mri.get(split, {}).get("malignant", []):
            copy_patient(entry, pos_dir, dry_run)
            info["classes"]["positive"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "fast_mri_malignant"
            })
            counts[split]["positive"] += 1

        # ── NEGATIVE: fastMRI benign ──
        for entry in fast_mri.get(split, {}).get("benign", []):
            copy_patient(entry, neg_dir, dry_run)
            info["classes"]["negative"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "fast_mri_benign"
            })
            counts[split]["negative"] += 1

        # ── NEGATIVE: fastMRI normal (true negatives) ──
        for entry in fast_mri.get(split, {}).get("normal", []):
            copy_patient(entry, neg_dir, dry_run)
            info["classes"]["negative"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "fast_mri_normal"
            })
            counts[split]["negative"] += 1

        # ── NEGATIVE: BreastDx normals ──
        for entry in breastdx_splits.get(split, []):
            copy_patient(entry, neg_dir, dry_run)
            info["classes"]["negative"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "breastdx_normal"
            })
            counts[split]["negative"] += 1

    # ── Summary ──
    print("\n  Patient counts per split:")
    print(f"  {'Split':<10} {'Positive':>10} {'Negative':>10} {'Total':>8} {'Ratio':>8}")
    print(f"  {'-'*50}")
    total_pos = total_neg = 0
    for split in ("train", "val", "test"):
        p = counts[split]["positive"]
        n = counts[split]["negative"]
        total_pos += p
        total_neg += n
        ratio = f"{p/n:.2f}" if n > 0 else "inf"
        print(f"  {split:<10} {p:>10} {n:>10} {p+n:>8} {ratio:>8}")
    print(f"  {'TOTAL':<10} {total_pos:>10} {total_neg:>10} {total_pos+total_neg:>8} "
          f"{total_pos/total_neg:.2f}" if total_neg > 0 else "")

    # Compute pos_weight for BCEWithLogitsLoss
    pos_weight = total_pos / total_neg if total_neg > 0 else 1.0
    info["pos_weight_for_bce"] = round(pos_weight, 4)
    info["counts_per_split"] = counts
    print(f"\n  BCEWithLogitsLoss pos_weight = {pos_weight:.4f}")
    print(f"  → In training: pos_weight = torch.tensor([{pos_weight:.4f}])")

    # Save dataset info JSON
    if not dry_run:
        json_path = os.path.join(STAGE1_OUT, "dataset_info.json")
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\n  Saved: {json_path}")

    return info


# ──────────────────────────────────────────────────────────
#  BUILD STAGE 3  (malignant vs benign)
# ──────────────────────────────────────────────────────────

def build_stage3(combined, fast_mri, dry_run=False):
    print("\n" + "="*60)
    print("  BUILDING STAGE 3 — Malignancy Classification")
    print("  Malignant: patients_combined + fastMRI malignant")
    print("  Benign:    fastMRI benign only")
    print("="*60)

    info   = {"stage": 3, "classes": {"malignant": [], "benign": []}}
    counts = {"train": {"malignant": 0, "benign": 0},
              "val":   {"malignant": 0, "benign": 0},
              "test":  {"malignant": 0, "benign": 0}}

    for split in ("train", "val", "test"):
        mal_dir = os.path.join(STAGE3_OUT, split, "malignant")
        ben_dir = os.path.join(STAGE3_OUT, split, "benign")
        if not dry_run:
            os.makedirs(mal_dir, exist_ok=True)
            os.makedirs(ben_dir, exist_ok=True)

        # ── MALIGNANT: patients_combined ──
        for entry in combined.get(split, []):
            copy_patient(entry, mal_dir, dry_run)
            info["classes"]["malignant"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "patients_combined"
            })
            counts[split]["malignant"] += 1

        # ── MALIGNANT: fastMRI malignant ──
        for entry in fast_mri.get(split, {}).get("malignant", []):
            copy_patient(entry, mal_dir, dry_run)
            info["classes"]["malignant"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "fast_mri_malignant"
            })
            counts[split]["malignant"] += 1

        # ── BENIGN: fastMRI benign ──
        for entry in fast_mri.get(split, {}).get("benign", []):
            copy_patient(entry, ben_dir, dry_run)
            info["classes"]["benign"].append({
                "patient_id": entry["patient_id"],
                "split": split,
                "source": "fast_mri_benign"
            })
            counts[split]["benign"] += 1

    # ── Summary ──
    print("\n  Patient counts per split:")
    print(f"  {'Split':<10} {'Malignant':>12} {'Benign':>8} {'Total':>8} {'Ratio':>8}")
    print(f"  {'-'*52}")
    total_mal = total_ben = 0
    for split in ("train", "val", "test"):
        m = counts[split]["malignant"]
        b = counts[split]["benign"]
        total_mal += m
        total_ben += b
        ratio = f"{m/b:.2f}" if b > 0 else "inf"
        print(f"  {split:<10} {m:>12} {b:>8} {m+b:>8} {ratio:>8}")
    print(f"  {'TOTAL':<10} {total_mal:>12} {total_ben:>8} {total_mal+total_ben:>8} "
          f"{total_mal/total_ben:.2f}" if total_ben > 0 else "")

    pos_weight = total_mal / total_ben if total_ben > 0 else 1.0
    info["pos_weight_for_bce"] = round(pos_weight, 4)
    info["counts_per_split"] = counts
    print(f"\n  BCEWithLogitsLoss pos_weight = {pos_weight:.4f}")
    print(f"  → In training: pos_weight = torch.tensor([{pos_weight:.4f}])")

    if not dry_run:
        json_path = os.path.join(STAGE3_OUT, "dataset_info.json")
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"\n  Saved: {json_path}")

    return info


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build classification datasets")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would be done without copying files")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage3-only", action="store_true")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("\n  [DRY RUN] No files will be copied.\n")

    # ── Check source folders exist ──
    missing = []
    for path, name in [
        (PATIENTS_COMBINED, "patients_combined"),
        (FAST_MRI_ROOT,     "fast_mri_complete"),
        (BREASTDX_NORMAL,   "breastdx_normal"),
    ]:
        if not os.path.isdir(path):
            missing.append(f"  MISSING: {path}  ({name})")
    if missing:
        print("\n[ERROR] The following source folders were not found:")
        for m in missing:
            print(m)
        print("\nEdit the PATHS section at the top of this script.")
        sys.exit(1)

    # ── Scan sources ──
    print("\n[1] Scanning source datasets...")

    combined      = scan_patients_combined(PATIENTS_COMBINED)
    fast_mri      = scan_fast_mri(FAST_MRI_ROOT)
    breastdx_sp   = split_flat_patients(BREASTDX_NORMAL, BREASTDX_SPLIT, RANDOM_SEED)

    # Print counts
    combined_total = sum(len(v) for v in combined.values())
    fm_total = {cls: sum(len(fast_mri[sp].get(cls, [])) for sp in fast_mri)
                for cls in ("benign", "malignant", "normal")}
    bdx_total = sum(len(v) for v in breastdx_sp.values())

    print(f"\n  patients_combined : {combined_total} malignant patients")
    print(f"    train={len(combined.get('train',[]))}  "
          f"val={len(combined.get('val',[]))}  "
          f"test={len(combined.get('test',[]))}")

    print(f"\n  fast_mri_complete :")
    for cls, n in fm_total.items():
        print(f"    {cls:<12}: {n} patients")

    print(f"\n  breastdx_normal   : {bdx_total} patients → will split "
          f"train={len(breastdx_sp['train'])} "
          f"val={len(breastdx_sp['val'])} "
          f"test={len(breastdx_sp['test'])}")

    # ── Build datasets ──
    print("\n[2] Building classification datasets...")

    if not args.stage3_only:
        s1_info = build_stage1(combined, fast_mri, breastdx_sp, dry_run)

    if not args.stage1_only:
        s3_info = build_stage3(combined, fast_mri, dry_run)

    # ── Final summary ──
    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    if not dry_run:
        if not args.stage3_only:
            print(f"\n  Stage 1 dataset  → {STAGE1_OUT}")
            print(f"    train/positive/   train/negative/")
            print(f"    val/positive/     val/negative/")
            print(f"    test/positive/    test/negative/")
        if not args.stage1_only:
            print(f"\n  Stage 3 dataset  → {STAGE3_OUT}")
            print(f"    train/malignant/  train/benign/")
            print(f"    val/malignant/    val/benign/")
            print(f"    test/malignant/   test/benign/")
    else:
        print("\n  Re-run without --dry-run to actually copy files.")

    print("\n  Next steps:")
    print("  1. Run with --dry-run first to verify counts")
    print("  2. Run without --dry-run to copy files")
    print("  3. Use dataset_info.json pos_weight in your BCEWithLogitsLoss")
    print("  4. Train Stage 1 → 3D ResNet18 on classification_stage1/")
    print("  5. Train Stage 3 → 3D EfficientNet-B0 on classification_stage3/")


if __name__ == "__main__":
    main()