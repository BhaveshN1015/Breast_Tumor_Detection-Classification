"""
Multi-model threshold sweep.
Scans ALL saved model folders, runs inference on the test set
for each model, finds the best threshold, and prints a full
ranking table so you know exactly which model to use.

Usage:
    python src/segmentation/find_best_threshold.py
"""
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

# -----------------------------
# PATHS
# -----------------------------
TEST_IMAGES = "data/Segmentation_dataset/test/images"
GT_DIR      = "data/Segmentation_dataset/test/masks"

# Root folder where all your model subfolders live
MODELS_ROOT = "models"

# Thresholds to try for each model
THRESHOLDS  = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# FIND ALL MODEL FILES
# Searches both direct .pth files and subfolders containing .pth files
# -----------------------------
def find_all_models(root):
    """
    Returns list of (display_name, full_path_to_pth_file).
    Handles:
      - root/filename.pth          → direct .pth files
      - root/subfolder/any.pth     → .pth inside named folders
    """
    found = []

    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)

        # Direct .pth file at root level
        if os.path.isfile(full) and entry.endswith(".pth"):
            found.append((entry.replace(".pth", ""), full))

        # Subfolder — look for .pth files inside
        elif os.path.isdir(full):
            pth_files = [f for f in os.listdir(full) if f.endswith(".pth")]
            for pth in sorted(pth_files):
                display = f"{entry}/{pth.replace('.pth','')}"
                found.append((display, os.path.join(full, pth)))

    return found


# -----------------------------
# LOAD MODEL
# Tries 3-channel first, falls back to 1-channel for old models
# -----------------------------
def load_model(pth_path, in_channels):
    m = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = in_channels,
        classes         = 1,
        activation      = None
    ).to(device)
    m.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    m.eval()
    return m


def try_load_model(pth_path):
    """Try loading as 3-channel first, then 1-channel for older models."""
    for ch in [3, 1]:
        try:
            m = load_model(pth_path, ch)
            return m, ch
        except RuntimeError:
            continue
    return None, None


# -----------------------------
# PREPROCESS IMAGE
# Handles both 3-channel and 1-channel
# -----------------------------
def preprocess(image_np, in_channels):
    img = cv2.resize(image_np, (256, 256))
    if in_channels == 3:
        # image_np is already (H,W,3) BGR
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))           # (3,256,256)
        tensor = torch.tensor(img).unsqueeze(0).float().to(device)
    else:
        # Grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor


# -----------------------------
# LOAD TEST DATA (once, reuse for all models)
# -----------------------------
print("\nLoading test images and masks...")

img_files = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(".png")])

test_images_3ch = []   # for 3-channel models
test_images_1ch = []   # for 1-channel models
test_gts        = []
test_names      = []

for img_name in tqdm(img_files, desc="Loading"):
    img_path  = os.path.join(TEST_IMAGES, img_name)
    mask_name = img_name.replace(".png", "_mask.png")
    gt_path   = os.path.join(GT_DIR, mask_name)

    if not os.path.exists(gt_path):
        continue

    img_3ch = cv2.imread(img_path, cv2.IMREAD_COLOR)     # (H,W,3)
    img_1ch = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # (H,W)
    gt      = cv2.imread(gt_path,  cv2.IMREAD_GRAYSCALE)

    if img_3ch is None or gt is None:
        continue

    gt = cv2.resize(gt, (256, 256))
    gt = (gt > 0).astype(np.uint8)

    test_images_3ch.append(img_3ch)
    test_images_1ch.append(img_1ch)
    test_gts.append(gt)
    test_names.append(img_name)

print(f"Loaded {len(test_gts)} test images")
print(f"Tumor slices: {sum(g.sum() > 0 for g in test_gts)}")


# -----------------------------
# DICE HELPER
# -----------------------------
def dice_np(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return (2 * inter) / (pred_bin.sum() + gt_bin.sum() + 1e-8)


# -----------------------------
# EVALUATE ONE MODEL
# -----------------------------
def evaluate_model(model, in_channels):
    """Run inference and return prob_maps list."""
    prob_maps = []
    images = test_images_3ch if in_channels == 3 else test_images_1ch

    with torch.no_grad():
        for img in images:
            tensor = preprocess(img, in_channels)
            out    = model(tensor)
            prob   = torch.sigmoid(out).squeeze().cpu().numpy()
            prob_maps.append(prob)

    return prob_maps


def sweep_thresholds(prob_maps):
    """Run threshold sweep and return best result."""
    results = []
    best_dice = 0
    best_t    = 0.1

    for t in THRESHOLDS:
        tumor_dices = []
        missed      = 0
        total_tp = total_fp = total_fn = 0

        for prob, gt in zip(prob_maps, test_gts):
            pred_bin  = (prob > t).astype(np.uint8)
            has_tumor = gt.sum() > 0

            if has_tumor:
                d = dice_np(pred_bin, gt)
                tumor_dices.append(d)
                if pred_bin.sum() == 0:
                    missed += 1

            total_tp += np.logical_and(pred_bin, gt).sum()
            total_fp += np.logical_and(pred_bin, 1 - gt).sum()
            total_fn += np.logical_and(1 - pred_bin, gt).sum()

        avg_dice  = np.mean(tumor_dices) if tumor_dices else 0
        recall    = total_tp / (total_tp + total_fn + 1e-8)
        precision = total_tp / (total_tp + total_fp + 1e-8)

        results.append((t, avg_dice, missed, recall, precision))

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_t    = t

    return results, best_dice, best_t


# -----------------------------
# MAIN SWEEP
# -----------------------------
all_models = find_all_models(MODELS_ROOT)

if not all_models:
    print("No .pth files found in models/ folder.")
    exit()

print(f"\nFound {len(all_models)} model files:")
for name, path in all_models:
    print(f"  {name:40s} → {path}")

print("\n" + "=" * 80)
print("  FULL MODEL COMPARISON")
print("=" * 80)
print(f"{'Model':<40} {'Ch':>3} {'Best Dice':>10} {'Best T':>7} {'Missed':>7}")
print("-" * 80)

all_results = []

for model_name, pth_path in all_models:

    model, in_ch = try_load_model(pth_path)

    if model is None:
        print(f"  {model_name:<38} FAILED TO LOAD — skipping")
        continue

    try:
        prob_maps = evaluate_model(model, in_ch)
        sweep, best_dice, best_t = sweep_thresholds(prob_maps)

        # find missed at best threshold
        best_missed = next(m for (t,d,m,r,p) in sweep if t == best_t)

        print(f"  {model_name:<38} {in_ch:>3}ch  {best_dice:>9.4f}  {best_t:>6.2f}  {best_missed:>6}")
        all_results.append((model_name, pth_path, in_ch, best_dice, best_t, best_missed, sweep))

    except Exception as e:
        print(f"  {model_name:<38} ERROR: {e}")

    # Free GPU memory between models
    del model
    torch.cuda.empty_cache()


# -----------------------------
# FINAL RANKING
# -----------------------------
print("\n" + "=" * 80)
print("  FINAL RANKING — by Tumor Dice")
print("=" * 80)

all_results.sort(key=lambda x: x[3], reverse=True)

for rank, (name, path, ch, dice, t, missed, sweep) in enumerate(all_results, 1):
    marker = " ← BEST" if rank == 1 else ""
    print(f"  #{rank:2d}  {name:<40} Dice={dice:.4f}  T={t:.2f}  Missed={missed}{marker}")

if all_results:
    best = all_results[0]
    print(f"\nWINNER: {best[0]}")
    print(f"  Path           : {best[1]}")
    print(f"  Channels       : {best[2]}")
    print(f"  Best Dice      : {best[3]:.4f}")
    print(f"  Best Threshold : {best[4]}")
    print(f"  Missed tumors  : {best[5]}")
    print(f"\nUpdate in predict_segmentation.py:")
    print(f"  MODEL_PATH = '{best[1]}'")
    print(f"  THRESHOLD  = {best[4]}")
    print(f"  in_channels = {best[2]}")

    # Print threshold sweep for the winner
    print(f"\nThreshold sweep for winner ({best[0]}):")
    print(f"{'Threshold':>12} | {'Dice':>10} | {'Missed':>8} | {'Recall':>8} | {'Precision':>10}")
    print("-" * 60)
    for (t, d, m, r, p) in best[6]:
        marker = " ← BEST" if t == best[4] else ""
        print(f"{t:>12.2f} | {d:>10.4f} | {m:>8} | {r:>8.4f} | {p:>10.4f}{marker}")