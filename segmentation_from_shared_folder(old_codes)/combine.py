"""
combine_and_split.py
====================
Combines old preprocessed + new preprocessed patients into a fresh
stratified train / val / test split and creates physical folders.

What it does:
  1. Scans ALL patients in data/patients_preprocessed/ (old + new)
  2. Creates a stratified split keeping old/new balanced in each split
  3. Creates physical folder structure:
       data/patients_combined/
           train/  <patient_id>/  image.npy  label.npy
           val/    <patient_id>/  image.npy  label.npy
           test/   <patient_id>/  image.npy  label.npy
  4. Files are COPIED — originals in patients_preprocessed/ untouched
  5. Writes data/dataset_3d_combined.json pointing to new folders
  6. Prints full summary and verifies all files exist

Run from project root:
    python combine_and_split.py
"""

import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────
SOURCE_ROOT = "data/patients_preprocessed"   # where old + new .npy files live
OUTPUT_ROOT = "data/patients_combined"        # NEW folder created here
OUTPUT_JSON = "data/dataset_3d_combined.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO = 0.15  (remainder)

RANDOM_SEED = 42
OLD_MAX_ID  = 100    # IDs 1–100 = old dataset,  101+ = new ISPY-1
# ──────────────────────────────────────────────────────────────────


def scan_all_patients(root):
    """
    Walk all split subfolders in SOURCE_ROOT.
    Collect every patient that has both image.npy and label.npy.
    Returns list of dicts with source paths.
    """
    patients = []
    missing  = []

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            print(f"  [WARN] Split folder not found: {split_dir}")
            continue

        for pid in sorted(
            os.listdir(split_dir),
            key=lambda x: int(x) if x.isdigit() else x
        ):
            pdir     = os.path.join(split_dir, pid)
            img_path = os.path.join(pdir, "image.npy")
            lbl_path = os.path.join(pdir, "label.npy")

            if not os.path.isdir(pdir):
                continue

            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                missing.append(f"{split}/{pid}")
                continue

            try:
                group = "old" if int(pid) <= OLD_MAX_ID else "new"
            except ValueError:
                group = "new"

            patients.append({
                "patient_id" : pid,
                "group"      : group,
                "img_src"    : img_path,
                "lbl_src"    : lbl_path,
            })

    if missing:
        print(f"  [WARN] {len(missing)} patients skipped (missing npy):")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")

    return patients


def stratified_split(patients, train_r, val_r, seed):
    """
    Split stratified by group so old and new patients are proportionally
    represented in every split. Returns (train, val, test) lists.
    """
    rng = np.random.default_rng(seed)

    old_pts = [p for p in patients if p["group"] == "old"]
    new_pts = [p for p in patients if p["group"] == "new"]

    def split_one_group(pts):
        idx     = rng.permutation(len(pts))
        n       = len(pts)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        return (
            [pts[i] for i in idx[:n_train]],
            [pts[i] for i in idx[n_train : n_train + n_val]],
            [pts[i] for i in idx[n_train + n_val :]],
        )

    old_tr, old_v, old_te = split_one_group(old_pts)
    new_tr, new_v, new_te = split_one_group(new_pts)

    def shuffle_combine(a, b):
        combined = a + b
        idx = rng.permutation(len(combined))
        return [combined[i] for i in idx]

    return (
        shuffle_combine(old_tr, new_tr),
        shuffle_combine(old_v,  new_v),
        shuffle_combine(old_te, new_te),
    )


def verify_no_overlap(train, val, test):
    """Check no patient_id appears in more than one split."""
    tr_ids = {p["patient_id"] for p in train}
    va_ids = {p["patient_id"] for p in val}
    te_ids = {p["patient_id"] for p in test}
    dupes  = (tr_ids & va_ids) | (tr_ids & te_ids) | (va_ids & te_ids)
    if dupes:
        print(f"  [ERROR] Patients in multiple splits: {dupes}")
        return False
    return True


def copy_patients_to_split(split_name, patients_list, output_root):
    """
    Copy image.npy + label.npy for every patient into:
        output_root / split_name / patient_id / image.npy
                                               / label.npy

    Skips patients whose destination already exists (resume support).
    Returns list of JSON-ready entry dicts.
    """
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    entries  = []
    skipped  = 0
    copied   = 0

    for p in tqdm(patients_list,
                  desc=f"  {split_name:5s}",
                  ncols=80,
                  leave=True):
        pid     = p["patient_id"]
        dst_dir = os.path.join(split_dir, pid)
        dst_img = os.path.join(dst_dir, "image.npy")
        dst_lbl = os.path.join(dst_dir, "label.npy")

        if os.path.exists(dst_img) and os.path.exists(dst_lbl):
            skipped += 1
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(p["img_src"], dst_img)
            shutil.copy2(p["lbl_src"], dst_lbl)
            copied += 1

        entries.append({
            "patient_id" : pid,
            "group"      : p["group"],
            "image"      : dst_img.replace("\\", "/"),
            "label"      : dst_lbl.replace("\\", "/"),
        })

    print(f"    copied={copied}  already_existed={skipped}")
    return entries


def verify_output(train_e, val_e, test_e):
    """Check every expected file actually exists on disk."""
    errors = 0
    for entries in [train_e, val_e, test_e]:
        for e in entries:
            for key in ["image", "label"]:
                if not os.path.exists(e[key]):
                    print(f"  MISSING: {e[key]}")
                    errors += 1
    return errors


def print_split_summary(name, pts):
    old = sum(1 for p in pts if p["group"] == "old")
    new = sum(1 for p in pts if p["group"] == "new")
    print(f"  {name:6s} : {len(pts):4d} total  "
          f"(old={old:3d}  new={new:3d})")


def main():
    print("=" * 62)
    print("  Combine + Split")
    print(f"  Source : {SOURCE_ROOT}")
    print(f"  Output : {OUTPUT_ROOT}/train  val  test")
    print(f"  JSON   : {OUTPUT_JSON}")
    print(f"  Seed   : {RANDOM_SEED}  |  "
          f"Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/"
          f"{int((1-TRAIN_RATIO-VAL_RATIO)*100)}")
    print("=" * 62)

    # ── 1. Scan ──────────────────────────────────────────────────
    print("\nStep 1 — Scanning source patients...")
    patients  = scan_all_patients(SOURCE_ROOT)
    old_count = sum(1 for p in patients if p["group"] == "old")
    new_count = sum(1 for p in patients if p["group"] == "new")

    print(f"  Total usable : {len(patients)}")
    print(f"  Old (≤ {OLD_MAX_ID})  : {old_count}")
    print(f"  New (> {OLD_MAX_ID})  : {new_count}")

    if len(patients) == 0:
        print("\n  ERROR: 0 patients found. "
              "Check SOURCE_ROOT and that image.npy/label.npy exist.")
        return

    # ── 2. Split ─────────────────────────────────────────────────
    print("\nStep 2 — Stratified split...")
    train_pts, val_pts, test_pts = stratified_split(
        patients, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    print()
    print_split_summary("TRAIN", train_pts)
    print_split_summary("VAL",   val_pts)
    print_split_summary("TEST",  test_pts)
    print(f"  {'TOTAL':6s} : "
          f"{len(train_pts)+len(val_pts)+len(test_pts):4d}")

    # ── 3. Overlap check ─────────────────────────────────────────
    print("\nStep 3 — Overlap check...")
    if not verify_no_overlap(train_pts, val_pts, test_pts):
        print("  Aborting.")
        return
    print("  No overlap. OK.")

    # ── 4. Copy files into new folder structure ───────────────────
    print(f"\nStep 4 — Copying into {OUTPUT_ROOT}/")
    print("  (Originals in patients_preprocessed/ are NOT touched)\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    train_entries = copy_patients_to_split("train", train_pts, OUTPUT_ROOT)
    val_entries   = copy_patients_to_split("val",   val_pts,   OUTPUT_ROOT)
    test_entries  = copy_patients_to_split("test",  test_pts,  OUTPUT_ROOT)

    # ── 5. Write JSON ────────────────────────────────────────────
    print(f"\nStep 5 — Writing {OUTPUT_JSON}...")

    # Strip 'group' from JSON entries (not needed at training time)
    def clean(entries):
        return [{"patient_id": e["patient_id"],
                 "image": e["image"],
                 "label": e["label"]}
                for e in entries]

    dataset = {
        "description"  : ("Combined old (1–100) + new ISPY-1 (101+) "
                          "breast DCE-MRI dataset"),
        "modality"     : {"0": "P1_early", "1": "P2_mid", "2": "P3_late"},
        "labels"       : {"0": "background", "1": "tumor"},
        "num_train"    : len(train_entries),
        "num_val"      : len(val_entries),
        "num_test"     : len(test_entries),
        "old_patients" : old_count,
        "new_patients" : new_count,
        "random_seed"  : RANDOM_SEED,
        "output_root"  : OUTPUT_ROOT.replace("\\", "/"),
        "training"     : clean(train_entries),
        "validation"   : clean(val_entries),
        "test"         : clean(test_entries),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Saved.")

    # ── 6. Verify ────────────────────────────────────────────────
    print("\nStep 6 — Verifying all files on disk...")
    errors = verify_output(train_entries, val_entries, test_entries)
    if errors == 0:
        print(f"  All {len(train_entries)+len(val_entries)+len(test_entries)}patients verified. OK.")
    else:
        print(f"  {errors} missing files — re-run to retry.")

    # ── Final summary ────────────────────────────────────────────
    total = len(train_entries) + len(val_entries) + len(test_entries)
    print()
    print("=" * 62)
    print("  Done")
    print("=" * 62)
    print(f"  data/patients_combined/")
    print(f"  ├── train/  {len(train_entries):4d} patients  "
          f"(old={sum(1 for e in train_entries if int(e['patient_id'])<=OLD_MAX_ID)}"
          f"  new={sum(1 for e in train_entries if int(e['patient_id'])>OLD_MAX_ID)})")
    print(f"  ├── val/    {len(val_entries):4d} patients  "
          f"(old={sum(1 for e in val_entries if int(e['patient_id'])<=OLD_MAX_ID)}"
          f"  new={sum(1 for e in val_entries if int(e['patient_id'])>OLD_MAX_ID)})")
    print(f"  └── test/   {len(test_entries):4d} patients  "
          f"(old={sum(1 for e in test_entries if int(e['patient_id'])<=OLD_MAX_ID)}"
          f"  new={sum(1 for e in test_entries if int(e['patient_id'])>OLD_MAX_ID)})")
    print(f"  Total       {total:4d} patients")
    print()
    print("  Next steps:")
    print("  1. In train_3d.py change:")
    print(f"       PREPROCESSED_ROOT = '{OUTPUT_ROOT}'")
    print("  2. In generate_dataset_json.py change:")
    print(f"       PATIENTS_ROOT = '{OUTPUT_ROOT}'")
    print("     Then run: python generate_dataset_json.py")
    print("  3. Start training on RTX 2000 Ada")


if __name__ == "__main__":
    main()