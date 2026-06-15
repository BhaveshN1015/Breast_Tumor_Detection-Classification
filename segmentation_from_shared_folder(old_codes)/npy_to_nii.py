import numpy as np
import nibabel as nib
import os

# 👉 SET YOUR PATH HERE
base_path = r"D:\Breast_Tumor_AI_Project\data\patients_preprocessed\train\2"

image_path = os.path.join(base_path, "image.npy")
label_path = os.path.join(base_path, "label.npy")

image = np.load(image_path)
label = np.load(label_path)

# Take first channel
img = image[0]

img_nifti = nib.Nifti1Image(img, affine=np.eye(4))
label_nifti = nib.Nifti1Image(label[0], affine=np.eye(4))

# Save in same folder
nib.save(img_nifti, os.path.join(base_path, "image.nii.gz"))
nib.save(label_nifti, os.path.join(base_path, "label.nii.gz"))

print("✅ Conversion done!")