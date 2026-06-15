import os
import shutil
import pandas as pd
import re
import nibabel as nib
import numpy as np

# ==============================
# PATHS
# ==============================

BASE_DIR = r"D:\Breast_Tumor_AI_Project\BreastDCEDL_spy1"
DCE_DIR = os.path.join(BASE_DIR, "spt1_dce")
MASK_DIR = os.path.join(BASE_DIR, "spy1_mask")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# HELPER
# ==============================

def extract_patient_id(filename):
    match = re.search(r"(ISPY1_\d+)", filename)
    return match.group(1) if match else None

# ==============================
# COLLECT DATA
# ==============================

patient_data = {}

# Images
for file in os.listdir(DCE_DIR):
    if not file.endswith((".nii", ".nii.gz")):
        continue

    pid = extract_patient_id(file)
    if pid is None:
        continue

    patient_data.setdefault(pid, {"images": [], "mask": None})
    patient_data[pid]["images"].append(os.path.join(DCE_DIR, file))

# Masks
for file in os.listdir(MASK_DIR):
    if not file.endswith((".nii", ".nii.gz")):
        continue

    pid = extract_patient_id(file)
    if pid is None:
        continue

    patient_data.setdefault(pid, {"images": [], "mask": None})
    patient_data[pid]["mask"] = os.path.join(MASK_DIR, file)

# ==============================
# PROCESS + VALIDATION
# ==============================

records = []

for pid, data in patient_data.items():
    patient_folder = os.path.join(OUTPUT_DIR, pid)
    os.makedirs(patient_folder, exist_ok=True)

    image_shapes = []
    valid_alignment = True

    # Copy images
    for i, img_path in enumerate(sorted(data["images"])):
        img = nib.load(img_path)
        shape = img.get_fdata().shape
        image_shapes.append(shape)

        shutil.copy(img_path, os.path.join(patient_folder, f"image_acq{i}.nii.gz"))

    # Process mask
    mask_present = data["mask"] is not None
    tumor_present = False
    mask_shape = None

    if mask_present:
        mask_img = nib.load(data["mask"])
        mask_data = mask_img.get_fdata()
        mask_shape = mask_data.shape

        shutil.copy(data["mask"], os.path.join(patient_folder, "mask.nii.gz"))

        # Check tumor
        if np.sum(mask_data) > 0:
            tumor_present = True

        # Check alignment
        for s in image_shapes:
            if s != mask_shape:
                valid_alignment = False

    else:
        valid_alignment = False

    # Final classification
    if not mask_present:
        status = "no_mask"
    elif not tumor_present:
        status = "empty_mask"
    elif not valid_alignment:
        status = "misaligned"
    else:
        status = "valid_tumor"

    records.append({
        "patient_id": pid,
        "num_phases": len(data["images"]),
        "mask_present": mask_present,
        "tumor_present": tumor_present,
        "mask_shape": str(mask_shape),
        "image_shape": str(image_shapes[0]) if image_shapes else "None",
        "alignment_ok": valid_alignment,
        "status": status
    })

# ==============================
# SAVE REPORT
# ==============================

df = pd.DataFrame(records)

excel_path = os.path.join(OUTPUT_DIR, "dataset_verified.xlsx")

try:
    df.to_excel(excel_path, index=False)
except:
    df.to_csv(excel_path.replace(".xlsx", ".csv"), index=False)

print("\n✅ DATASET FULLY VERIFIED!")
print("📊 Report saved at:", excel_path)

# ==============================
# SUMMARY PRINT
# ==============================

print("\nSummary:")
print(df["status"].value_counts())