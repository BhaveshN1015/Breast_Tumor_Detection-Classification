import torch
import torch.nn as nn

# Simulate exactly what train_3d.py does
# Fake prediction (what model outputs before sigmoid) - all zeros = untrained model
batch_size = 2
pred  = torch.zeros(batch_size, 1, 96, 96, 96)   # raw logits from untrained model
label = torch.zeros(batch_size, 1, 96, 96, 96)   # all background

# Put tumor in label (simulate a tumor patch)
label[0, 0, 40:55, 40:55, 40:55] = 1.0

print("=== LABEL CHECK ===")
print(f"Label dtype        : {label.dtype}")
print(f"Label unique vals  : {torch.unique(label)}")
print(f"Label tumor voxels : {(label > 0).sum().item()}")

print("\n=== LOSS CHECK ===")

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-6):
    pred_sig = torch.sigmoid(pred)
    TP = (pred_sig * target).sum()
    FP = (pred_sig * (1 - target)).sum()
    FN = ((1 - pred_sig) * target).sum()
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky

def dice_bce_loss(pred, target, smooth=1e-6):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum()
    dice = 1 - (2 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
    bce  = nn.functional.binary_cross_entropy_with_logits(pred, target)
    return dice + bce

def combined_loss(pred, target):
    return tversky_loss(pred, target) + dice_bce_loss(pred, target)

loss = combined_loss(pred, label)
print(f"Combined loss value : {loss.item():.6f}")
print(f"Loss is nan        : {torch.isnan(loss).item()}")
print(f"Loss is inf        : {torch.isinf(loss).item()}")

# Check what sigmoid of zero logits gives
pred_sig = torch.sigmoid(pred)
print(f"\n=== SIGMOID OUTPUT (untrained model) ===")
print(f"Sigmoid min/max    : {pred_sig.min():.4f} / {pred_sig.max():.4f}")
print(f"All values are 0.5 : {(pred_sig == 0.5).all().item()}")

# Check dice at threshold 0.5 for untrained model
pred_bin = (pred_sig > 0.5).float()
print(f"\n=== DICE AT THRESHOLD 0.5 (untrained) ===")
print(f"Predicted positives: {pred_bin.sum().item()}")
print(f"  (untrained model outputs 0.5 everywhere, threshold 0.5 = nothing predicted)")

pred_bin_01 = (pred_sig > 0.1).float()
print(f"\n=== DICE AT THRESHOLD 0.1 (untrained) ===")
intersection = (pred_bin_01 * label).sum()
dice = (2 * intersection) / (pred_bin_01.sum() + label.sum() + 1e-6)
print(f"Predicted positives: {pred_bin_01.sum().item()}")
print(f"Dice               : {dice.item():.6f}")
print(f"  (threshold 0.1 catches everything since sigmoid(0)=0.5 > 0.1)")

print("\n=== CONCLUSION ===")
if not torch.isnan(loss) and not torch.isinf(loss):
    print("Loss computation is fine.")
    print(">>> The issue is likely one of these in train_3d.py:")
    print("    1. Training not running enough epochs to learn")
    print("    2. LR too low / warmup not working")
    print("    3. Validation using threshold 0.5 instead of 0.1")
    print("    4. dataset_3d.json has wrong paths or too few patients")