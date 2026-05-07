"""
predict_3d.py
=============

Sliding window inference on preprocessed .npy patient volumes.

Changes from previous version:
  - Loads data via NpyDataset (data_3d_simplified) — no NIfTI loading
  - Saves output masks as .nii.gz with identity affine (no original NIfTI needed)
  - THRESHOLD = 0.7  (optimal modal threshold from combined training run)
  - autocast uses torch.amp namespace
  - num_workers=0, pin_memory=False  (Windows stability)

Usage:
    python src/segmentation_3d/predict_3d.py
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from scipy import ndimage
from tqdm import tqdm

from monai.inferers import sliding_window_inference

sys.path.append(os.path.dirname(__file__))

from data_3d  import NpyDataset, get_val_transforms, PATCH_SIZE, _build_data_list
from model_3d import get_model
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

PREPROCESSED_ROOT = "data/patients_combined"
MODEL_PATH        = "models/segmentation_3d/epochs/BEST_RAW_80th_epoch_dice0.6850.pth"
OUTPUT_DIR        = "outputs/segmentation_3d_predictions"
THRESHOLD         = 0.8      # optimal from training — use modal threshold from training run
TTA_ENABLED       = True
SW_OVERLAP        = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA GPU not found."
device = torch.device("cuda")
torch.cuda.set_device(0)

print(f"Using device: {device}")
print(f"  GPU  : {torch.cuda.get_device_name(0)}")
print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ------------------------------------------------------------------ #
#  LOAD MODEL                                                          #
# ------------------------------------------------------------------ #

model = get_model(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()
print(f"\nModel loaded: {MODEL_PATH}")


# ------------------------------------------------------------------ #
#  SINGLE FORWARD PASS                                                 #
# ------------------------------------------------------------------ #

def predict_volume(image_tensor):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(
                inputs        = image_tensor,
                roi_size      = PATCH_SIZE,
                sw_batch_size = 4,
                predictor     = model,
                overlap       = SW_OVERLAP,
                mode          = "gaussian",
            )
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------ #
#  TTA — 8 flip orientations                                           #
# ------------------------------------------------------------------ #

def predict_with_tta(image_tensor):
    img_np   = image_tensor.squeeze(0).cpu().numpy()   # (3, D, H, W)
    prob_sum = predict_volume(image_tensor)

    flip_combos = [
        (2,), (3,), (4,),
        (2, 3), (2, 4), (3, 4),
        (2, 3, 4),
    ]

    for axes in flip_combos:
        flipped_np     = np.flip(img_np, axis=[a - 1 for a in axes]).copy()
        flipped_t      = torch.tensor(flipped_np).unsqueeze(0).float().to(device)
        prob_flip      = predict_volume(flipped_t)
        prob_flip_back = np.flip(prob_flip, axis=[a - 2 for a in axes]).copy()
        prob_sum      += prob_flip_back

    return (prob_sum / (1 + len(flip_combos))).astype(np.float32)


# ------------------------------------------------------------------ #
#  POST-PROCESSING                                                     #
# ------------------------------------------------------------------ #

def post_process(binary_mask, min_size_voxels=200):
    struct  = ndimage.generate_binary_structure(3, 1)
    closed  = ndimage.binary_closing(binary_mask, structure=struct, iterations=2)
    labeled, num_features = ndimage.label(closed)

    if num_features == 0:
        return closed.astype(np.uint8)

    cleaned = np.zeros_like(closed, dtype=np.uint8)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() >= min_size_voxels:
            cleaned[labeled == i] = 1
    return cleaned

def keep_largest_component(binary_mask):
    """
    Keep only the single largest connected component.
    Breast tumors are almost always one connected mass.
    Removes all satellite blobs regardless of size.
    """
    labeled, n_features = ndimage.label(binary_mask)
    if n_features == 0:
        return binary_mask
    # Find largest component
    sizes = ndimage.sum(binary_mask, labeled, range(1, n_features + 1))
    largest_label = np.argmax(sizes) + 1
    result = (labeled == largest_label).astype(np.uint8)
    return result


# ------------------------------------------------------------------ #
#  MAIN INFERENCE LOOP                                                 #
# ------------------------------------------------------------------ #

test_list   = _build_data_list(PREPROCESSED_ROOT, "test")
test_ds     = NpyDataset(test_list, transform=get_val_transforms())
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=0, pin_memory=False)

print(f"\nRunning inference on {len(test_ds)} test patients...")
print(f"  TTA       : {'8 orientations' if TTA_ENABLED else 'disabled'}")
print(f"  Threshold : {THRESHOLD}")
print(f"  Output    : {OUTPUT_DIR}")
print()

results_summary = []

for idx, batch in enumerate(tqdm(test_loader, desc="Predicting", ncols=80)):
    images = batch["image"].to(device)
    labels = batch["label"]

    patient_id = test_list[idx]["patient_id"]

    if TTA_ENABLED:
        prob_map = predict_with_tta(images)
    else:
        prob_map = predict_volume(images)

    binary_mask = (prob_map > THRESHOLD).astype(np.uint8)
    binary_mask = post_process(binary_mask, min_size_voxels=200)
    binary_mask = keep_largest_component(binary_mask) 

    # Save as NIfTI with identity affine (preprocessed data has no original affine)
    out_name = f"patient{patient_id}_pred_mask.nii.gz"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    nib.save(nib.Nifti1Image(binary_mask, np.eye(4)), out_path)

    gt_np   = (labels.squeeze().numpy() > 0).astype(bool)
    pred_np = binary_mask.astype(bool)
    inter   = np.logical_and(pred_np, gt_np).sum()
    dice    = (2 * inter) / (pred_np.sum() + gt_np.sum() + 1e-8)

    results_summary.append({
        "patient"        : patient_id,
        "dice"           : round(float(dice), 4),
        "has_tumor"      : bool(gt_np.sum() > 0),
        "predicted_tumor": bool(pred_np.sum() > 0),
    })

print(f"\nPredictions saved to: {OUTPUT_DIR}")
print("\nQuick summary:")
tumor_dices = [r["dice"] for r in results_summary if r["has_tumor"]]
missed      = sum(1 for r in results_summary if r["has_tumor"] and not r["predicted_tumor"])
print(f"  Test patients     : {len(results_summary)}")
print(f"  Tumor patients    : {sum(r['has_tumor'] for r in results_summary)}")
print(f"  Missed tumors     : {missed}")
if tumor_dices:
    print(f"  Mean tumor Dice   : {np.mean(tumor_dices):.4f}")
    print(f"  Best Dice         : {np.max(tumor_dices):.4f}")
print()
print("Next step: python src/segmentation_3d/evaluate_3d.py")