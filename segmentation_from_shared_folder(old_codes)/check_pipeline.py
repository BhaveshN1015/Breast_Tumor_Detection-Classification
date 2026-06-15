import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, SpacingD, OrientationD,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd,
    CropForegroundd, ResizeWithPadOrCropd,
    ConcatItemsd, EnsureTyped
)

# Use your actual patient 1 path
data_item = {
    "image_0": r"D:\Breast_Tumor_AI_Project\data\patients\train\1\P1.nii.gz",
    "image_1": r"D:\Breast_Tumor_AI_Project\data\patients\train\1\P2.nii.gz",
    "image_2": r"D:\Breast_Tumor_AI_Project\data\patients\train\1\P3.nii.gz",
    "label"  : r"D:\Breast_Tumor_AI_Project\data\patients\train\1\GT.nii.gz",
}

transforms = Compose([
    LoadImaged(
        keys=["image_0","image_1","image_2","label"],
        image_only=True, ensure_channel_first=True
    ),
    SpacingD(
        keys=["image_0","image_1","image_2","label"],
        pixdim=(1.0,1.0,1.0),
        mode=("bilinear","bilinear","bilinear","nearest"),
    ),
    OrientationD(
        keys=["image_0","image_1","image_2","label"],
        axcodes="RAS",
    ),
    ScaleIntensityRangePercentilesd(
        keys=["image_0","image_1","image_2"],
        lower=0.5, upper=99.5,
        b_min=0.0, b_max=1.0, clip=True,
    ),
    NormalizeIntensityd(
        keys=["image_0","image_1","image_2"],
        nonzero=True, channel_wise=True,
    ),
])

print("Running transforms step by step...")
result = transforms(data_item)

img_shape = result["image_0"].shape
lbl_shape = result["label"].shape
tumor_voxels = (result["label"] > 0).sum().item()

print(f"\nAfter Spacing+Orientation:")
print(f"  Image shape  : {img_shape}")
print(f"  Label shape  : {lbl_shape}")
print(f"  Tumor voxels : {tumor_voxels}")

# Now test CropForeground with image_0 as source
from monai.transforms import CropForegroundd
crop = CropForegroundd(
    keys=["image_0","image_1","image_2","label"],
    source_key="image_0"
)
result2 = crop(result)
img2 = result2["image_0"].shape
lbl2 = result2["label"].shape
tumor2 = (result2["label"] > 0).sum().item()

print(f"\nAfter CropForeground (source=image_0):")
print(f"  Image shape  : {img2}")
print(f"  Label shape  : {lbl2}")
print(f"  Tumor voxels : {tumor2}")

# Check if tumor survives resize
from monai.transforms import ResizeWithPadOrCropd
resize = ResizeWithPadOrCropd(
    keys=["image_0","image_1","image_2","label"],
    spatial_size=(192,192,192)
)
result3 = resize(result2)
tumor3 = (result3["label"] > 0).sum().item()

print(f"\nAfter ResizeWithPadOrCrop (192,192,192):")
print(f"  Image shape  : {result3['image_0'].shape}")
print(f"  Tumor voxels : {tumor3}")

if tumor3 == 0:
    print("  *** TUMOR LOST AFTER RESIZE — THIS IS THE BUG ***")
else:
    print(f"  Tumor survived OK")