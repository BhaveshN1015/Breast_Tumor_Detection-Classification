import os
import pandas as pd

# =========================
# CONFIG
# =========================
DATA_ROOT = "data/patients_combined"
METADATA_CSV = r"D:\Breast_Tumor_AI_Project\data\BreastDCEDL_spy1_metadata.csv"
OUTPUT_FILE = "final_patient_labels.xlsx"

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(METADATA_CSV)

# Create mapping: patient_id -> label
label_map = {}

for _, row in df.iterrows():
    pid = row["pid"]  # e.g., ISPY1_1001

    try:
        # Extract numeric part
        num = int(pid.split("_")[-1])  # 1001
        mapped_id = num - 900          # 1001 → 101

    except:
        continue

    # Classification logic
    if row.get("TripleNeg", 0) == 1 or row.get("HER2pos", 0) == 1:
        label = "malignant"
    elif row.get("HRposHER2neg", 0) == 1:
        label = "benign"
    else:
        label = "unknown"

    label_map[str(mapped_id)] = label

# =========================
# SCAN DATASET
# =========================
results = []

for split in ["train", "val", "test"]:
    split_path = os.path.join(DATA_ROOT, split)

    if not os.path.exists(split_path):
        continue

    for patient_id in os.listdir(split_path):
        patient_path = os.path.join(split_path, patient_id)

        if not os.path.isdir(patient_path):
            continue

        pid_int = int(patient_id)

        # OLD DATASET
        if pid_int <= 100:
            label = "old_dataset_skip"

        # NEW DATASET (ISPY)
        else:
            label = label_map.get(patient_id, "unknown")

        results.append({
            "patient_id": patient_id,
            "split": split,
            "label": label
        })

# =========================
# SAVE EXCEL
# =========================
out_df = pd.DataFrame(results)
out_df = out_df.sort_values(by=["split", "patient_id"])

out_df.to_excel(OUTPUT_FILE, index=False)

print(f"✅ Done! Labels saved to {OUTPUT_FILE}")