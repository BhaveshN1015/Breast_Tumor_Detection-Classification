"""
dataset_stage1.py
==================
Dataset for Stage 1 — Tumor Detection (binary 0/1).

FIXES vs previous versions:
  v1 → MetaTensor crash (fixed: .as_tensor() after transforms)
  v2 → Shape mismatch crash (fixed: pad+crop to FIXED size)

ROOT CAUSE OF v2 CRASH:
  SpatialPadd only pads small volumes up to a minimum — it does NOT
  shrink larger ones. So fastMRI (3×141×213×213) and existing patients
  (3×217×128×128) came out different sizes → torch.stack() crashed.

THE FIX:
  After SpatialPadd (which pads small volumes up), apply
  CenterSpatialCropd (which crops large volumes down).
  Together they guarantee every volume exits as exactly FIXED_SIZE,
  regardless of the input dimensions.

FIXED_SIZE = (128, 192, 192)  — D × H × W
  RTX 3050 VRAM at batch=2: ~1.6 GB  ✓
  RTX 2000 Ada at batch=4:  ~3.0 GB  ✓

Folder structure:
  data/classification_stage1/
    train/  val/  test/
      negative/ {pid}/ image.npy
      positive/ {pid}/ image.npy
"""

import os
import platform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    SpatialPadd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

# ── Windows: always use 0 workers to avoid spawn issues ───────────
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4

# ── FIXED output size — every volume is resized to this exactly ───
# SpatialPadd pads smaller volumes UP to this size (zero-padding).
# CenterSpatialCropd crops larger volumes DOWN to this size.
# Result: all outputs are EXACTLY (3, 128, 192, 192).
FIXED_SIZE = (128, 192, 192)   # (D, H, W)


def _to_plain_tensor(x):
    """
    Strip MONAI MetaTensor metadata → plain torch.Tensor float32.
    This prevents the 'Trying to resize storage not resizable' crash.
    """
    if hasattr(x, "as_tensor"):
        return x.as_tensor().float().contiguous()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).float()
    elif isinstance(x, torch.Tensor):
        return x.float().contiguous()
    return torch.tensor(x, dtype=torch.float32)


class Stage1Dataset(Dataset):
    """
    Returns:
      image : float32 tensor  shape (3, 128, 192, 192)  — fixed, stackable
      label : float32 scalar  0.0 = negative / 1.0 = positive
      path  : str             source image path (for debugging)
    """

    def __init__(self, root_dir, split="train", augment=True,
                 fixed_size=FIXED_SIZE):
        super().__init__()
        self.split      = split
        self.fixed_size = fixed_size
        self.samples    = []   # list of (img_path, label_int)

        split_dir = os.path.join(root_dir, split)
        for cls_name, cls_label in [("negative", 0), ("positive", 1)]:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"  [WARN] directory not found: {cls_dir}")
                continue
            for pid in sorted(os.listdir(cls_dir)):
                pid_dir = os.path.join(cls_dir, pid)
                if not os.path.isdir(pid_dir):
                    continue
                img_path = os.path.join(pid_dir, "image.npy")
                if os.path.exists(img_path):
                    self.samples.append((img_path, cls_label))

        n_pos = sum(1 for _, l in self.samples if l == 1)
        n_neg = sum(1 for _, l in self.samples if l == 0)
        print(f"  Stage1Dataset [{split:5s}]: {len(self.samples):3d} patients"
              f"  pos={n_pos}  neg={n_neg}")

        # ── Core resize transforms (applied to EVERY sample) ──────
        # Step 1: pad volumes that are SMALLER than fixed_size
        # Step 2: crop volumes that are LARGER than fixed_size
        # After both steps every volume is EXACTLY fixed_size.
        self.resize_tf = Compose([
            SpatialPadd(
                keys=["image"],
                spatial_size=fixed_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(
                keys=["image"],
                roi_size=fixed_size,
            ),
        ])

        # ── Augmentation transforms (train split only) ────────────
        if augment and split == "train":
            self.aug_tf = Compose([
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3,
                              max_k=3, spatial_axes=(1, 2)),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            ])
        else:
            self.aug_tf = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_label = self.samples[idx]

        # 1. Load numpy (3, D, H, W)
        image = np.load(img_path, allow_pickle=False).astype(np.float32)

        # 2. Pad then crop → exactly fixed_size
        data = self.resize_tf({"image": image})

        # 3. Augment (train only)
        if self.aug_tf is not None:
            data = self.aug_tf(data)

        # 4. Strip MetaTensor → plain torch.Tensor  (v1 fix)
        image_tensor = _to_plain_tensor(data["image"])

        # 5. Sanity check (removed in production, useful during debug)
        assert image_tensor.shape == (3, *self.fixed_size), (
            f"Shape mismatch: expected (3,{self.fixed_size}), "
            f"got {tuple(image_tensor.shape)}")

        return {
            "image" : image_tensor,   # (3, 128, 192, 192) always
            "label" : torch.tensor(cls_label, dtype=torch.float32),
            "path"  : img_path,
        }


def get_dataloaders(root_dir, batch_size=2, num_workers=None,
                    fixed_size=FIXED_SIZE):
    """Returns (train_loader, val_loader, test_loader) for Stage 1."""
    if num_workers is None:
        num_workers = _DEFAULT_WORKERS

    train_ds = Stage1Dataset(root_dir, "train", augment=True,  fixed_size=fixed_size)
    val_ds   = Stage1Dataset(root_dir, "val",   augment=False, fixed_size=fixed_size)
    test_ds  = Stage1Dataset(root_dir, "test",  augment=False, fixed_size=fixed_size)

    persist = num_workers > 0

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=persist)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                               shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                               shuffle=False, **kw)

    return train_loader, val_loader, test_loader
