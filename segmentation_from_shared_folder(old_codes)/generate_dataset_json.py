"""
generate_dataset_json.py
========================

Scans data/patients_preprocessed/train|val|test folders and generates
dataset_3d.json for the simplified data_3d.py pipeline.

Since prepare_3d_offline.py saved each patient as:
    image.npy  (3, D, H, W) — P1+P2+P3 already stacked
    label.npy  (1, D, H, W) — GT mask

The JSON format is simpler than before — no separate P1/P2/P3 paths.
Each entry is just {"image": "path/image.npy", "label": "path/label.npy"}.

Run:
    python src/segmentation_3d/generate_dataset_json.py

Output: data/dataset_3d.json
"""

import os
import json

# ------------------------------------------------------------------ #
#  PATHS                                                               #
# ------------------------------------------------------------------ #

PATIENTS_ROOT = "data/patients_combined"   # preprocessed .npy files
OUTPUT_JSON   = "data/dataset_3d.json"

SPLITS = {
    "training"  : os.path.join(PATIENTS_ROOT, "train"),
    "validation": os.path.join(PATIENTS_ROOT, "val"),
    "test"      : os.path.join(PATIENTS_ROOT, "test"),
}


def build_split(split_path, split_name):
    """
    Walk every patient subfolder and build a list of
    {"image": "path/image.npy", "label": "path/label.npy"} dicts.
    Skips patients where either file is missing.
    """
    if not os.path.exists(split_path):
        print(f"  [WARNING] Split folder not found: {split_path}")
        return []

    patients = sorted(
        [p for p in os.listdir(split_path)
         if os.path.isdir(os.path.join(split_path, p))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    entries = []
    skipped = []

    for pid in patients:
        patient_dir = os.path.join(split_path, pid)
        image_path  = os.path.join(patient_dir, "image.npy")
        label_path  = os.path.join(patient_dir, "label.npy")

        if not os.path.exists(image_path):
            print(f"  [SKIP] patient {pid}: image.npy not found")
            skipped.append(pid)
            continue

        if not os.path.exists(label_path):
            print(f"  [SKIP] patient {pid}: label.npy not found")
            skipped.append(pid)
            continue

        entries.append({
            "image": image_path,
            "label": label_path,
        })

    print(f"  {split_name}: {len(entries)} patients added"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return entries


# ------------------------------------------------------------------ #
#  BUILD JSON                                                          #
# ------------------------------------------------------------------ #

print("=" * 55)
print("  Generating 3D dataset JSON  (preprocessed .npy)")
print("=" * 55)
print(f"  Source : {PATIENTS_ROOT}")
print(f"  Output : {OUTPUT_JSON}")
print()

dataset = {}

for split_key, split_path in SPLITS.items():
    dataset[split_key] = build_split(split_path, split_key)

dataset["description"] = "Breast Tumor DCE-MRI 3D segmentation (preprocessed)"
dataset["modality"]    = {"0": "P1_early", "1": "P2_mid", "2": "P3_late"}
dataset["labels"]      = {"0": "background", "1": "tumor"}
dataset["numTraining"] = len(dataset["training"])
dataset["format"]      = "npy"

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset, f, indent=2)

print()
print("=" * 55)
print("  dataset_3d.json created successfully")
print("=" * 55)
print(f"  Training patients  : {len(dataset['training'])}")
print(f"  Validation patients: {len(dataset['validation'])}")
print(f"  Test patients      : {len(dataset['test'])}")
print(f"  Saved to           : {OUTPUT_JSON}")
print()
print("  Next step: python src/segmentation_3d/train_3d.py")