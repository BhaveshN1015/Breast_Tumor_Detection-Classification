import os
import pydicom
import numpy as np
import nibabel as nib
from tqdm import tqdm

# 🔧 INPUT PATH (your dataset)
INPUT_DIR = r"F:\unknowns"

# 🔧 OUTPUT PATH (where .nii.gz will be saved)
OUTPUT_DIR = r"F:\converted_nii"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ✅ Function to load and stack DICOM slices
def load_dicom_series(folder_path):
    series_dict = {}

    # 🔍 Collect DICOM files grouped by SeriesInstanceUID
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                path = os.path.join(root, file)
                try:
                    dcm = pydicom.dcmread(path)
                    series_uid = dcm.SeriesInstanceUID

                    if series_uid not in series_dict:
                        series_dict[series_uid] = []

                    series_dict[series_uid].append(dcm)
                except:
                    continue

    if len(series_dict) == 0:
        return None

    # 🔥 Pick the largest series (MOST IMPORTANT FIX)
    selected_series = max(series_dict.values(), key=len)

    slices = selected_series

    # ✅ Sort properly
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            print("⚠️ Sorting issue")

    # ✅ Ensure same shape
    valid_slices = []
    base_shape = slices[0].pixel_array.shape

    for s in slices:
        if s.pixel_array.shape == base_shape:
            valid_slices.append(s)

    # 📦 Stack
    images = [s.pixel_array for s in valid_slices]
    volume = np.stack(images, axis=0)

    return volume, valid_slices


# ✅ Convert one patient
def convert_patient(patient_path, patient_id):
    result = load_dicom_series(patient_path)

    if result is None:
        print(f"❌ No DICOM found in {patient_id}")
        return

    volume, slices = result

    # 📐 Get spacing (important for NIfTI)
    try:
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = slices[0].SliceThickness
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness))
    except:
        spacing = (1.0, 1.0, 1.0)

    # 🧭 Create affine matrix
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1])

    # 💾 Save as NIfTI
    nii_image = nib.Nifti1Image(volume, affine)

    output_path = os.path.join(OUTPUT_DIR, f"{patient_id}.nii.gz")
    nib.save(nii_image, output_path)

    print(f"✅ Saved: {output_path} | Shape: {volume.shape}")


# 🔁 Loop through all patients
def main():
    patients = [p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))]

    print(f"🔍 Found {len(patients)} patients")

    for patient in tqdm(patients):
        patient_path = os.path.join(INPUT_DIR, patient)
        convert_patient(patient_path, patient)


if __name__ == "__main__":
    main()