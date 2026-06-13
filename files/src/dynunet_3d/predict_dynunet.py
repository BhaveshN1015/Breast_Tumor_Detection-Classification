"""
predict_dynunet.py
==================

Sliding window inference on preprocessed .npy patient volumes using DynUNet.

Changes from previous version:
  - Loads data via NpyDataset — no NIfTI loading, no _expand_paths
  - Saves output masks as .nii.gz with identity affine
  - THRESHOLD = 0.7  (optimal modal threshold from combined training run)
  - autocast uses torch.amp namespace
  - num_workers=0, pin_memory=False  (Windows stability)

Usage:
    python src/segmentation_3d/predict_dynunet.py
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
from torch.amp import autocast
from scipy import ndimage
from tqdm import tqdm

from monai.inferers import sliding_window_inference

sys.path.append(os.path.dirname(__file__))

from data_3d       import NpyDataset, get_val_transforms, PATCH_SIZE, _build_data_list
from model_dynunet import get_dynunet
from torch.utils.data import DataLoader

PREPROCESSED_ROOT = "data/patients_combined"
MODEL_PATH        = "models/dynunet_3d/dynunet_best_raw.pth"
OUTPUT_DIR        = "outputs/dynunet_predictions"
THRESHOLD         = 0.7       # optimal modal threshold from training run
TTA_ENABLED       = True
SW_OVERLAP        = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = get_dynunet(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print(f"\nModel loaded: {MODEL_PATH}")


def predict_volume(image_tensor):
    with torch.no_grad():
        with autocast("cuda"):
            logits = sliding_window_inference(
                inputs=image_tensor, roi_size=PATCH_SIZE,
                sw_batch_size=4, predictor=model,
                overlap=SW_OVERLAP, mode="gaussian",
            )
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


def predict_with_tta(image_tensor):
    img_np   = image_tensor.squeeze(0).cpu().numpy()
    prob_sum = predict_volume(image_tensor)
    for axes in [(2,),(3,),(4,),(2,3),(2,4),(3,4),(2,3,4)]:
        flipped_np     = np.flip(img_np, axis=[a-1 for a in axes]).copy()
        flipped_t      = torch.tensor(flipped_np).unsqueeze(0).float().to(device)
        prob_flip      = predict_volume(flipped_t)
        prob_sum      += np.flip(prob_flip, axis=[a-2 for a in axes]).copy()
    return (prob_sum / 8).astype(np.float32)


def post_process(binary_mask, min_size_voxels=200):
    struct  = ndimage.generate_binary_structure(3, 1)
    closed  = ndimage.binary_closing(binary_mask, structure=struct, iterations=2)
    labeled, n = ndimage.label(closed)
    if n == 0:
        return closed.astype(np.uint8)
    cleaned = np.zeros_like(closed, dtype=np.uint8)
    for i in range(1, n + 1):
        if (labeled == i).sum() >= min_size_voxels:
            cleaned[labeled == i] = 1
    return cleaned


def keep_largest_component(binary_mask):
    """Keep only the single largest connected component — removes satellite blobs."""
    labeled, n_features = ndimage.label(binary_mask)
    if n_features == 0:
        return binary_mask
    sizes = ndimage.sum(binary_mask, labeled, range(1, n_features + 1))
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(np.uint8)


test_list   = _build_data_list(PREPROCESSED_ROOT, "test")
test_ds     = NpyDataset(test_list, transform=get_val_transforms())
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=0, pin_memory=False)

print(f"\nRunning inference on {len(test_ds)} test patients...")
print(f"  TTA: {'8 orientations' if TTA_ENABLED else 'disabled'}  |  "
      f"Threshold: {THRESHOLD}  |  Output: {OUTPUT_DIR}\n")

results_summary = []

for idx, batch in enumerate(tqdm(test_loader, desc="Predicting", ncols=80)):
    images     = batch["image"].to(device)
    labels     = batch["label"]
    patient_id = test_list[idx]["patient_id"]

    prob_map    = predict_with_tta(images) if TTA_ENABLED else predict_volume(images)
    binary_mask = post_process((prob_map > THRESHOLD).astype(np.uint8))
    binary_mask = keep_largest_component(binary_mask)

    nib.save(nib.Nifti1Image(binary_mask, np.eye(4)),
             os.path.join(OUTPUT_DIR, f"patient{patient_id}_dynunet_pred.nii.gz"))

    gt_np   = (labels.squeeze().numpy() > 0).astype(bool)
    pred_np = binary_mask.astype(bool)
    inter   = np.logical_and(pred_np, gt_np).sum()
    dice    = (2 * inter) / (pred_np.sum() + gt_np.sum() + 1e-8)

    results_summary.append({
        "patient": patient_id, "dice": round(float(dice), 4),
        "has_tumor": bool(gt_np.sum() > 0),
        "predicted_tumor": bool(pred_np.sum() > 0),
    })

tumor_dices = [r["dice"] for r in results_summary if r["has_tumor"]]
missed      = sum(1 for r in results_summary if r["has_tumor"] and not r["predicted_tumor"])

print(f"\nPredictions saved to: {OUTPUT_DIR}")
print(f"  Test patients : {len(results_summary)}")
print(f"  Tumor patients: {sum(r['has_tumor'] for r in results_summary)}")
print(f"  Missed tumors : {missed}")
if tumor_dices:
    print(f"  Mean Dice     : {np.mean(tumor_dices):.4f}")
    print(f"  Best Dice     : {np.max(tumor_dices):.4f}")
    print(f"  Worst Dice    : {np.min(tumor_dices):.4f}")
print("\nNext step: python src/segmentation_3d/ensemble_3model.py")