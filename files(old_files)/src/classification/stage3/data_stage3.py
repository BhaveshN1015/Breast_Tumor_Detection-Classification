"""
data_stage3.py
==============
DataLoader for Stage 3 — Benign vs Malignant classification.

FIXES vs previous version:
  ✓ pin_memory=False  (was True — crashes on Windows with MONAI/numpy loaders)
  ✓ persistent_workers=False when num_workers=0  (was True — crash on Windows)
  ✓ Everything else unchanged

Input:
  classification_stage3/{train|val|test}/{malignant|benign}/{pid}/
      image.npy   (3, D, H, W)  float32
      label.npy   (1, D, H, W)  float32  (segmentation mask, may be all-zeros)

Processing:
  1. Load image.npy and label.npy
  2. Find tumour centroid from label mask (or use volume centre if absent)
  3. Crop CROP_SIZE³ patch centred on centroid
  4. Augment (train only): random flips, intensity jitter, rot90
  5. Return plain torch.Tensor (3, 64, 64, 64) + scalar label
"""

import os
import platform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Windows-safe default workers ──────────────────────────────────
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4

# ── Config ────────────────────────────────────────────────────────
CROP_SIZE   = 64
STAGE3_ROOT = r"D:\Breast_Tumor_AI_Project\data\classification_stage3"


# ── Plain tensor conversion ────────────────────────────────────────
def _to_plain_tensor(x):
    """Convert ndarray or MetaTensor → plain float32 torch.Tensor."""
    if hasattr(x, "as_tensor"):
        return x.as_tensor().float().contiguous()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).float()
    elif isinstance(x, torch.Tensor):
        return x.float().contiguous()
    return torch.tensor(x, dtype=torch.float32)


# ── Centroid + crop helpers ────────────────────────────────────────

def _get_centroid(label_3d):
    """
    label_3d: (D, H, W) numpy float32
    Returns (z, y, x) int centroid. None if mask is empty.
    """
    if label_3d is None or label_3d.sum() == 0:
        return None
    coords = np.argwhere(label_3d > 0.5)
    return coords.mean(axis=0).astype(int)


def _crop_patch(img, label, crop_size):
    """
    img   : (3, D, H, W) float32 numpy
    label : (1, D, H, W) float32 numpy or None
    Returns (3, C, C, C) numpy patch guaranteed to be exactly crop_size³.
    """
    c = crop_size
    _, D, H, W = img.shape

    # Pad if any dim < crop_size
    pad_D = max(0, c - D)
    pad_H = max(0, c - H)
    pad_W = max(0, c - W)
    if pad_D > 0 or pad_H > 0 or pad_W > 0:
        img = np.pad(img,
                     ((0, 0),
                      (pad_D//2, pad_D - pad_D//2),
                      (pad_H//2, pad_H - pad_H//2),
                      (pad_W//2, pad_W - pad_W//2)),
                     mode="reflect")
        if label is not None:
            label = np.pad(label,
                           ((0, 0),
                            (pad_D//2, pad_D - pad_D//2),
                            (pad_H//2, pad_H - pad_H//2),
                            (pad_W//2, pad_W - pad_W//2)),
                           mode="constant", constant_values=0)

    _, D, H, W = img.shape

    # Centroid
    lbl_3d   = label[0] if label is not None else None
    centroid = _get_centroid(lbl_3d)
    if centroid is None:
        centroid = np.array([D // 2, H // 2, W // 2])

    z = int(np.clip(centroid[0], c // 2, D - c // 2))
    y = int(np.clip(centroid[1], c // 2, H - c // 2))
    x = int(np.clip(centroid[2], c // 2, W - c // 2))

    patch = img[:, z-c//2:z+c//2, y-c//2:y+c//2, x-c//2:x+c//2]

    # Fallback centre crop if shape wrong
    if patch.shape != (3, c, c, c):
        patch = img[:, D//2-c//2:D//2+c//2,
                       H//2-c//2:H//2+c//2,
                       W//2-c//2:W//2+c//2]

    return patch


# ── Dataset ───────────────────────────────────────────────────────

class Stage3Dataset(Dataset):
    """
    label: 1 = malignant   0 = benign
    Returns: plain float32 tensor (3,64,64,64) + scalar float32 label
    """

    def __init__(self, root, split, augment=False, crop_size=CROP_SIZE):
        self.augment   = augment
        self.crop_size = crop_size
        self.samples   = []   # (img_path, lbl_path_or_None, int_label)

        for cls_name, int_label in [("malignant", 1), ("benign", 0)]:
            cls_dir = os.path.join(root, split, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"  [WARN] not found: {cls_dir}")
                continue
            for pid in sorted(os.listdir(cls_dir)):
                img_p = os.path.join(cls_dir, pid, "image.npy")
                lbl_p = os.path.join(cls_dir, pid, "label.npy")
                if os.path.isfile(img_p):
                    self.samples.append((
                        img_p,
                        lbl_p if os.path.isfile(lbl_p) else None,
                        int_label,
                    ))

        n_mal = sum(1 for _, _, l in self.samples if l == 1)
        n_ben = sum(1 for _, _, l in self.samples if l == 0)
        print(f"  Stage3Dataset [{split:5s}]: {len(self.samples):3d} patients"
              f"  mal={n_mal}  ben={n_ben}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, label = self.samples[idx]

        img = np.load(img_path, allow_pickle=False).astype(np.float32)
        lbl = (np.load(lbl_path, allow_pickle=False).astype(np.float32)
               if lbl_path else None)

        patch = _crop_patch(img, lbl, self.crop_size)
        patch = np.clip(patch, -3.5, 7.0)

        if self.augment:
            patch = self._augment(patch)

        img_tensor = _to_plain_tensor(patch)
        return img_tensor, torch.tensor(label, dtype=torch.float32)

    def _augment(self, patch):
        for axis in (1, 2, 3):
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=axis).copy()
        for c in range(3):
            patch[c] = patch[c] * np.random.uniform(0.9, 1.1) + \
                       np.random.uniform(-0.05, 0.05)
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(2, 3)).copy()
        return patch


# ── Loader builders ───────────────────────────────────────────────
# FIX: pin_memory=False and persistent_workers=False when num_workers=0
# On Windows, pin_memory=True with numpy-based loaders causes crashes.
# persistent_workers=True requires num_workers > 0.

def build_loaders_stage3(root=STAGE3_ROOT, batch_size=16,
                          num_workers=None, crop_size=CROP_SIZE):
    """Standard DataLoader — no oversampling."""
    if num_workers is None:
        num_workers = _DEFAULT_WORKERS

    train_ds = Stage3Dataset(root, "train", augment=True,  crop_size=crop_size)
    val_ds   = Stage3Dataset(root, "val",   augment=False, crop_size=crop_size)
    test_ds  = Stage3Dataset(root, "test",  augment=False, crop_size=crop_size)

    # pin_memory=False — safe on Windows; persistent_workers only if workers > 0
    kw = dict(num_workers=num_workers, pin_memory=False,
              persistent_workers=(num_workers > 0))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   drop_last=True, **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def build_weighted_stage3(root=STAGE3_ROOT, batch_size=16,
                           num_workers=None, crop_size=CROP_SIZE,
                           malignant_weight=1.5):
    """WeightedRandomSampler — malignant drawn 1.5× more often."""
    if num_workers is None:
        num_workers = _DEFAULT_WORKERS

    train_ds = Stage3Dataset(root, "train", augment=True,  crop_size=crop_size)
    val_ds   = Stage3Dataset(root, "val",   augment=False, crop_size=crop_size)
    test_ds  = Stage3Dataset(root, "test",  augment=False, crop_size=crop_size)

    weights = [malignant_weight if l == 1 else 1.0
               for _, _, l in train_ds.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    kw = dict(num_workers=num_workers, pin_memory=False,
              persistent_workers=(num_workers > 0))

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                   drop_last=True, **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )
