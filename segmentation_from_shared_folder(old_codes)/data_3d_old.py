"""
data_3d.py  (SIMPLIFIED — loads preprocessed .npy files)
==========================================================

Use this AFTER running prepare_3d_offline.py.

Since all resampling, orientation, and normalisation was done offline,
_base_transforms() is now just 3 lines — load, ensure type, done.
Training epoch time drops from ~2.5 min to under 30 seconds.

Key settings based on actual data measurements:
  - Native volume: 896×896×120, spacing 0.379×0.379×1.70mm
  - After 1.5mm resample + breast crop: ~192×226×136 voxels
  - Tumor: 19,114 voxels, spans large portion of volume
  - Tumor/volume ratio: 0.0198% raw → ~0.5% after breast crop

PATCH_SIZE = (96, 96, 96)
  Correct for this dataset. After breast crop the volume is ~192×226×136.
  A 96³ patch captures ~half the volume in each dimension — enough
  context around the tumor. The tumor spans most of the Y and Z extent
  so nearly every patch will contain tumor voxels.

ratios=[3, 1] in RandCropByLabelClassesd:
  3 tumor-centred patches per 1 background patch.
  With 0.5% tumor ratio this keeps the model focused on tumor regions
  without completely ignoring background context.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAffined,
    EnsureTyped,
    ToTensord,
    SpatialPadd,
)

from monai.data import CacheDataset, DataLoader as MonaiLoader

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

PATCH_SIZE    = (96, 96, 96)
NUM_SAMPLES   = 6    # patches extracted per volume per step
VOXEL_SPACING = (1.5, 1.5, 1.5)  # informational only — already applied


# ------------------------------------------------------------------ #
#  CUSTOM DATASET — loads .npy files directly                        #
# ------------------------------------------------------------------ #

class NpyDataset(Dataset):
    """
    Loads preprocessed image.npy (3,D,H,W) and label.npy (1,D,H,W).
    Returns dict {"image": tensor, "label": tensor}.
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        image = np.load(entry["image"]).astype(np.float32)  # (3, D, H, W)
        label = np.load(entry["label"]).astype(np.float32)  # (1, D, H, W)

        item = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }

        if self.transform:
            item = self.transform(item)

        return item


# ------------------------------------------------------------------ #
#  TRANSFORMS                                                          #
# ------------------------------------------------------------------ #

def get_train_transforms():
    return Compose([
        # Guarantee volume is at least patch size before cropping
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=PATCH_SIZE,
            mode="reflect",
        ),

        # 3 tumour-centred patches for every 1 background patch
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            num_classes=2,
            num_samples=NUM_SAMPLES,
            ratios=[4, 1],
        ),

        # Spatial augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
        ),

        # Intensity augmentations
        RandShiftIntensityd(keys=["image"], offsets=0.1,  prob=0.4),
        RandScaleIntensityd(keys=["image"], factors=0.1,  prob=0.4),
        RandGaussianNoised(keys=["image"],  prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.2,
        ),
    ])


def get_val_transforms():
    """Deterministic — no augmentation. SpatialPad only."""
    return Compose([
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=PATCH_SIZE,
            mode="reflect",
        ),
    ])


# ------------------------------------------------------------------ #
#  DATASET BUILDER                                                     #
# ------------------------------------------------------------------ #

def _build_data_list(preprocessed_root, split):
    """
    Scans preprocessed_root/split/ for patient folders
    containing image.npy and label.npy.
    """
    split_dir = os.path.join(preprocessed_root, split)
    if not os.path.exists(split_dir):
        print(f"  [WARNING] Split not found: {split_dir}")
        return []

    patients = sorted(
        [p for p in os.listdir(split_dir)
         if os.path.isdir(os.path.join(split_dir, p))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    data_list = []
    for pid in patients:
        img_path = os.path.join(split_dir, pid, "image.npy")
        lbl_path = os.path.join(split_dir, pid, "label.npy")
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_list.append({"image": img_path, "label": lbl_path,
                               "patient_id": pid})
        else:
            print(f"  [SKIP] Patient {pid}: missing image.npy or label.npy")

    return data_list


def build_loaders(preprocessed_root="data/patients_preprocessed",
                  batch_size=2, num_workers=0):
    """
    Returns (train_loader, val_loader, test_loader).

    num_workers=0  — required on Windows (MONAI worker crashes)
    pin_memory=False — required on Windows (CUDA mapping error)
    """
    train_list = _build_data_list(preprocessed_root, "train")
    val_list   = _build_data_list(preprocessed_root, "val")
    test_list  = _build_data_list(preprocessed_root, "test")

    train_ds = NpyDataset(train_list, transform=get_train_transforms())
    val_ds   = NpyDataset(val_list,   transform=get_val_transforms())
    test_ds  = NpyDataset(test_list,  transform=get_val_transforms())

    print(f"[Dataset] Train : {len(train_ds)} patients")
    print(f"[Dataset] Val   : {len(val_ds)} patients")
    print(f"[Dataset] Test  : {len(test_ds)} patients")
    print(f"[Dataset] Patch : {PATCH_SIZE}  |  Patches/vol: {NUM_SAMPLES}")

    train_loader = MonaiLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = MonaiLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = MonaiLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader
