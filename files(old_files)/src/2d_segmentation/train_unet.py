import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from tqdm import tqdm
from torch.utils.data import DataLoader

from data import SegmentationDataset

# SOLUTION 6: pretrained ResNet34 encoder U-Net
# Install once: pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp


# -----------------------------
# Paths
# -----------------------------
DATA_DIR   = "data/Segmentation_dataset"
TRAIN_IMG  = os.path.join(DATA_DIR, "train/images")
TRAIN_MASK = os.path.join(DATA_DIR, "train/masks")
VAL_IMG    = os.path.join(DATA_DIR, "val/images")
VAL_MASK   = os.path.join(DATA_DIR, "val/masks")
SAVE_DIR   = "models"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# Augmentation
# -----------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=25, p=0.5),
    A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.GridDistortion(p=0.2),
])

val_transform = None


# -----------------------------
# Combined Loss: Tversky + Dice + BCE
#
# Tversky(alpha=0.3, beta=0.7): penalizes false negatives 2.3x more
# than false positives → pushes recall up toward precision
# Dice: ensures good overlap quality
# BCE: ensures pixel-level accuracy
# Together: best balance for small tumor detection
# -----------------------------
def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-6):
    pred = torch.sigmoid(pred)
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky


def dice_bce_loss(pred, target, smooth=1e-6):
    pred_sig = torch.sigmoid(pred)
    intersection = (pred_sig * target).sum()
    dice = 1 - (2 * intersection + smooth) / \
               (pred_sig.sum() + target.sum() + smooth)
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    return dice + bce


def combined_loss(pred, target):
    return tversky_loss(pred, target) + dice_bce_loss(pred, target)


# -----------------------------
# Metrics
# -----------------------------
def dice_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return ((2 * intersection + smooth) / (union + smooth)).item()


def compute_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    return precision.item(), recall.item(), accuracy.item()


# -----------------------------
# Training
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = SegmentationDataset(
        TRAIN_IMG, TRAIN_MASK, transform=train_transform, mode="train"
    )
    val_dataset = SegmentationDataset(
        VAL_IMG, VAL_MASK, transform=val_transform, mode="val"
    )

    # SOLUTION 5: batch size 16 — more stable gradient estimates
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # SOLUTION 6: ResNet34 U-Net with ImageNet pretrained encoder
    # in_channels=3  → 3-channel DCE-MRI (P1+P2+P3 stacked)
    # classes=1       → binary tumor mask
    # activation=None → we apply sigmoid manually in loss/metrics
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None
    ).to(device)

    print("Model: ResNet34 U-Net (pretrained ImageNet encoder)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    EPOCHS        = 40
    WARMUP_EPOCHS = 5

    # Start LR very low — warmup prevents destroying pretrained weights
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    def warmup_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS * (1e-4 / 1e-6)
        else:
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return (1e-4 / 1e-6) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    patience       = 30
    counter        = 0
    best_dice      = 0.0   # smoothed dice — used for early stopping
    best_raw_dice  = 0.0   # single-epoch dice — saves every genuine peak

    history = {
        "train_loss": [], "train_recall": [],
        "val_loss": [],   "val_dice": [],
        "val_precision": [], "val_recall": [], "val_accuracy": []
    }

    EVAL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

    for epoch in range(EPOCHS):

        current_lr = optimizer.param_groups[0]['lr']
        phase = "WARMUP" if epoch < WARMUP_EPOCHS else "TRAIN"
        print(f"\nEpoch {epoch+1}/{EPOCHS}  [{phase}]  LR: {current_lr:.2e}")

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        tp = fp = fn = tn = 0

        loop = tqdm(train_loader, desc="Training")

        for images, masks in loop:
            images = images.to(device)
            masks  = masks.to(device)

            preds = model(images)
            loss  = combined_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                p = torch.sigmoid(preds)
                p = (p > 0.5).float()
                tp += (p * masks).sum().item()
                fp += (p * (1 - masks)).sum().item()
                fn += ((1 - p) * masks).sum().item()
                tn += ((1 - p) * (1 - masks)).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_recall   = tp / (tp + fn + 1e-6)

        history["train_loss"].append(avg_train_loss)
        history["train_recall"].append(train_recall)

        print(f"Train Loss   : {avg_train_loss:.4f}")
        print(f"Train Recall : {train_recall:.4f}")

        # ---- VALIDATION ----
        model.eval()
        val_loss  = 0.0
        val_dices = {t: 0.0 for t in EVAL_THRESHOLDS}
        prec_sum = rec_sum = acc_sum = 0.0

        with torch.no_grad():

            val_loop = tqdm(val_loader, desc="Validation")

            for images, masks in val_loop:
                images = images.to(device)
                masks  = masks.to(device)

                preds = model(images)
                loss  = combined_loss(preds, masks)
                val_loss += loss.item()

                for t in EVAL_THRESHOLDS:
                    val_dices[t] += dice_score(preds, masks, threshold=t)

                p, r, a = compute_metrics(preds, masks, threshold=0.5)
                prec_sum += p
                rec_sum  += r
                acc_sum  += a

                val_loop.set_postfix(loss=f"{loss.item():.4f}")

        n_val        = len(val_loader)
        avg_val_loss = val_loss / n_val

        best_threshold = max(EVAL_THRESHOLDS, key=lambda t: val_dices[t])
        avg_val_dice   = val_dices[best_threshold] / n_val
        avg_prec       = prec_sum / n_val
        avg_rec        = rec_sum  / n_val
        avg_acc        = acc_sum  / n_val

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_precision"].append(avg_prec)
        history["val_recall"].append(avg_rec)
        history["val_accuracy"].append(avg_acc)

        print(f"Val Loss      : {avg_val_loss:.4f}")
        print(f"Val Dice      : {avg_val_dice:.4f}  (threshold={best_threshold})")
        print(f"Val Precision : {avg_prec:.4f}")
        print(f"Val Recall    : {avg_rec:.4f}")
        print(f"Val Accuracy  : {avg_acc:.4f}")
        for t in EVAL_THRESHOLDS:
            print(f"  Dice@{t} = {val_dices[t]/n_val:.4f}")

        scheduler.step()

        # Always save every epoch — you can pick the best one later
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_epoch_{epoch+1}.pth")

        # --- TRACK 1: Raw single-epoch Dice ---
        # Saves unet_best_raw.pth whenever any single epoch beats the record.
        # Catches good epochs like ep13 (0.7445) even after bad ep11,12.
        if avg_val_dice > best_raw_dice:
            best_raw_dice = avg_val_dice
            torch.save(model.state_dict(), f"{SAVE_DIR}/unet_best_raw.pth")
            print(f"  >>> Raw best saved    (Dice={best_raw_dice:.4f}, epoch={epoch+1})")

        # --- TRACK 2: Smoothed 3-epoch Dice (for early stopping) ---
        # More stable — prevents stopping on a single bad epoch.
        if len(history["val_dice"]) >= 3:
            smoothed_dice = np.mean(history["val_dice"][-3:])
        else:
            smoothed_dice = avg_val_dice

        if smoothed_dice > best_dice:
            best_dice = smoothed_dice
            counter   = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/unet_best.pth")
            print(f"  *** Smoothed best saved (smoothed Dice={best_dice:.4f}) ***")
        else:
            counter += 1
            print(f"  No improvement ({counter}/{patience})")

        if counter >= patience:
            print("Early stopping triggered.")
            break

        np.save("training_history.npy", history)

    print(f"\nTraining complete.")
    print(f"Best smoothed Dice : {best_dice:.4f}  → models/unet_best.pth")
    print(f"Best raw Dice      : {best_raw_dice:.4f}  → models/unet_best_raw.pth")
    print(f"All epoch models   : models/unet_epoch_1.pth ... unet_epoch_N.pth")
    print(f"Use unet_best_raw.pth for prediction — it has the highest single-epoch Dice")


if __name__ == "__main__":
    main()