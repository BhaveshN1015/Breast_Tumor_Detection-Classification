"""
data_3d.py  — upgraded for combined dataset (old + new ISPY-1)
==============================================================

Key upgrades from previous version:
  - NEW: WeightedRandomSampler — 2x upsampling of new-dataset patients
    (IDs > 100). Old patients seen once per epoch, new patients twice,
    so the model learns the transposed-axis geometry properly.
  - NEW: dataset_group tag ("old"/"new") on every data_list entry so
    predict_3d.py can use per-group thresholds at inference time.
  - NEW: build_weighted_loaders() for training.
    build_loaders() kept for inference/evaluation compatibility.
  - SpatialPadd mode "constant" → "reflect": reduces edge artefacts
    when padding the new dataset's thin D axis (~50 slices).
  - PATCH_SIZE stays (96,96,96) — correct for both geometries.
  - ratios=[4,1] kept — 4 tumour patches : 1 background patch.
  - NEW: Rand3DElasticd added to training transforms (prob=0.2)
    Helps the model generalise across old (D≈220) and new (D≈50)
    dataset geometry without overfitting to either axis layout.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from monai.transforms import (
    Compose,
    Rand3DElasticd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAffined,
    SpatialPadd,
)

from monai.data import DataLoader as MonaiLoader

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

PATCH_SIZE    = (96, 96, 96)
NUM_SAMPLES   = 6
VOXEL_SPACING = (1.5, 1.5, 1.5)

OLD_ID_CUTOFF = 100


# ------------------------------------------------------------------ #
#  CUSTOM DATASET                                                      #
# ------------------------------------------------------------------ #

class NpyDataset(Dataset):
    """
    Loads preprocessed image.npy (3,D,H,W) and label.npy (1,D,H,W).
    Returns dict with "image", "label", "patient_id", "dataset_group".
    dataset_group is "old" (IDs <= 100) or "new" (IDs > 100).
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]
        image = np.load(entry["image"]).astype(np.float32)
        label = np.load(entry["label"]).astype(np.float32)

        item = {
            "image"         : torch.from_numpy(image),
            "label"         : torch.from_numpy(label),
            "patient_id"    : entry["patient_id"],
            "dataset_group" : entry.get("dataset_group", "old"),
        }
        if self.transform:
            item = self.transform(item)
        return item


# ------------------------------------------------------------------ #
#  TRANSFORMS                                                          #
# ------------------------------------------------------------------ #

def get_train_transforms():
    return Compose([
        # reflect-pad to at least 96^3 — avoids zero-border artefacts
        # especially important for new dataset D~50 axis
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=PATCH_SIZE,
            mode="reflect",
        ),
        # 4 tumour-centred patches : 1 background patch
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
        # Elastic deformation — helps generalise across old (D~220) and
        # new ISPY-1 (D~50) dataset axis proportions. Kept mild (prob=0.2)
        # to avoid distorting small tumours.
        Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.2,
            sigma_range=(3, 5),
            magnitude_range=(50, 150),
            padding_mode="border",
        ),
        # Intensity augmentations
        RandShiftIntensityd(keys=["image"], offsets=0.1,  prob=0.4),
        RandScaleIntensityd(keys=["image"], factors=0.1,  prob=0.4),
        RandGaussianNoised(keys=["image"],  prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
            prob=0.2,
        ),
    ])


def get_val_transforms():
    """Deterministic — reflect-pad only, no augmentation."""
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
    Scans preprocessed_root/split/ for patient folders.
    Tags each entry with dataset_group "old"/"new".
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
    n_old = n_new = 0
    for pid in patients:
        img_path = os.path.join(split_dir, pid, "image.npy")
        lbl_path = os.path.join(split_dir, pid, "label.npy")
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            try:
                pid_int = int(pid)
                group   = "old" if pid_int <= OLD_ID_CUTOFF else "new"
            except ValueError:
                group = "old"
            if group == "old":
                n_old += 1
            else:
                n_new += 1
            data_list.append({
                "image"         : img_path,
                "label"         : lbl_path,
                "patient_id"    : pid,
                "dataset_group" : group,
            })
        else:
            print(f"  [SKIP] Patient {pid}: missing image.npy or label.npy")

    print(f"  [Dataset:{split}] old={n_old}  new={n_new}  total={len(data_list)}")
    return data_list


def _make_weighted_sampler(data_list, new_weight=2.0):
    """
    WeightedRandomSampler that samples new-dataset patients new_weight
    times more often than old-dataset patients.
    """
    weights = [
        new_weight if e.get("dataset_group", "old") == "new" else 1.0
        for e in data_list
    ]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(data_list),
        replacement=True,
    )
    n_old = sum(1 for e in data_list if e.get("dataset_group","old") == "old")
    n_new = len(data_list) - n_old
    print(f"  [Sampler] old={n_old} (w=1.0)  new={n_new} (w={new_weight})")
    return sampler


def build_weighted_loaders(preprocessed_root="data/patients_combined",
                           batch_size=4, num_workers=0,
                           new_dataset_weight=2.0):
    """
    Returns (train_loader, val_loader, test_loader).
    train_loader uses WeightedRandomSampler — new patients 2x sampled.
    Use this for all training runs on the combined dataset.
    """
    train_list = _build_data_list(preprocessed_root, "train")
    val_list   = _build_data_list(preprocessed_root, "val")
    test_list  = _build_data_list(preprocessed_root, "test")

    train_ds = NpyDataset(train_list, transform=get_train_transforms())
    val_ds   = NpyDataset(val_list,   transform=get_val_transforms())
    test_ds  = NpyDataset(test_list,  transform=get_val_transforms())

    print(f"\n[Dataset] Patch : {PATCH_SIZE}  |  Patches/vol: {NUM_SAMPLES}")

    train_sampler = _make_weighted_sampler(train_list, new_weight=new_dataset_weight)

    train_loader = MonaiLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
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


def build_loaders(preprocessed_root="data/patients_combined",
                  batch_size=4, num_workers=0):
    """
    Backward-compatible loader — no weighted sampling.
    Used by predict_3d.py / predict_dynunet.py / evaluate scripts.
    """
    train_list = _build_data_list(preprocessed_root, "train")
    val_list   = _build_data_list(preprocessed_root, "val")
    test_list  = _build_data_list(preprocessed_root, "test")

    train_ds = NpyDataset(train_list, transform=get_train_transforms())
    val_ds   = NpyDataset(val_list,   transform=get_val_transforms())
    test_ds  = NpyDataset(test_list,  transform=get_val_transforms())

    print(f"\n[Dataset] Patch : {PATCH_SIZE}  |  Patches/vol: {NUM_SAMPLES}")

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
