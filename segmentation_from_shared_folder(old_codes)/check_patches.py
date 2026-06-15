import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, SpacingD, OrientationD,
    ScaleIntensityRangePercentilesd, NormalizeIntensityd,
    CropForegroundd, ResizeWithPadOrCropd,
    ConcatItemsd, EnsureTyped,
    RandCropByLabelClassesd
)

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
    CropForegroundd(
        keys=["image_0","image_1","image_2","label"],
        source_key="image_0"
    ),
    ResizeWithPadOrCropd(
        keys=["image_0","image_1","image_2","label"],
        spatial_size=(192,192,192)
    ),
    ConcatItemsd(
        keys=["image_0","image_1","image_2"],
        name="image", dim=0
    ),
    EnsureTyped(keys=["image","label"], dtype=torch.float32),
])

print("Running full pipeline...")
result = transforms(data_item)

image = result["image"]
label = result["label"]

print(f"\n=== AFTER FULL PIPELINE ===")
print(f"Image shape       : {image.shape}")
print(f"Label shape       : {label.shape}")
print(f"Label unique vals : {torch.unique(label)}")
print(f"Label min/max     : {label.min():.4f} / {label.max():.4f}")
print(f"Tumor voxels      : {(label > 0).sum().item()}")
print(f"Image min/max     : {image.min():.4f} / {image.max():.4f}")

# Now test the patch sampler
print(f"\n=== TESTING PATCH SAMPLER ===")
sampler = RandCropByLabelClassesd(
    keys=["image","label"],
    label_key="label",
    spatial_size=(96,96,96),
    num_classes=2,
    num_samples=6,
    ratios=[1, 2],
)

patches = sampler(result)
print(f"Number of patches extracted : {len(patches)}")

tumor_counts = []
for i, patch in enumerate(patches):
    t = (patch["label"] > 0).sum().item()
    tumor_counts.append(t)
    print(f"  Patch {i+1}: image={patch['image'].shape}  tumor_voxels={t}")

print(f"\nPatches with tumor : {sum(1 for t in tumor_counts if t > 0)} / {len(tumor_counts)}")
print(f"Avg tumor voxels per patch : {np.mean(tumor_counts):.1f}")

if all(t == 0 for t in tumor_counts):
    print("\n*** SAMPLER IS NOT FINDING TUMOR — THIS IS THE BUG ***")
elif sum(1 for t in tumor_counts if t > 0) < 2:
    print("\n*** SAMPLER RARELY FINDS TUMOR — RATIOS NEED FIXING ***")
else:
    print("\nSampler is finding tumor OK")
    print(">>> Bug is likely in the training loss or label dtype")