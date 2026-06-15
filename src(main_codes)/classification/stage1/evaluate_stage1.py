"""
evaluate_stage1.py
===================
Evaluate Stage 1 on test set, or run single-patient inference.

Usage:
    # Full test set:
    python src/classification/stage1/evaluate_stage1.py

    # Single patient folder:
    python src/classification/stage1/evaluate_stage1.py \
        --patient_dir "D:\\...\\classification_stage1\\test\\positive\\fastMRI_breast_010"

    # Adjust threshold (default 0.5):
    python src/classification/stage1/evaluate_stage1.py --threshold 0.4
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from monai.transforms import Compose, SpatialPadd, CenterSpatialCropd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.classification.stage1.dataset_stage1 import (
    Stage1Dataset, FIXED_SIZE, _to_plain_tensor)
from src.classification.stage1.model_stage1 import build_model

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, confusion_matrix, classification_report)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

DATA_ROOT  = r"D:\Breast_Tumor_AI_Project\data\classification_stage1"
MODEL_PATH = (r"D:\Breast_Tumor_AI_Project\models"
              r"\classification_stage1\best_model.pth")


# ── Full test set evaluation ──────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(model, device, data_root, fixed_size, threshold):
    test_ds = Stage1Dataset(data_root, "test", augment=False,
                             fixed_size=fixed_size)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=0)

    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        label  = int(batch["label"].item())

        with autocast(device_type=device.type,
                      enabled=(device.type == "cuda")):
            logit = model(images)

        prob   = float(torch.sigmoid(logit).cpu().item())
        pred   = int(prob >= threshold)
        pid    = os.path.basename(os.path.dirname(batch["path"][0]))
        status = "✓" if pred == label else "✗"

        print(f"  {status} {pid:35s}  gt={label}  "
              f"pred={pred}  prob={prob:.4f}")

        all_labels.append(label)
        all_probs.append(prob)
        all_preds.append(pred)

    print("\n" + "─" * 65)

    if SKLEARN_OK and len(set(all_labels)) > 1:
        acc = sum(p == l for p, l in zip(all_preds, all_labels))
        acc = acc / len(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        f1  = f1_score(all_labels, all_preds, zero_division=0)
        cm  = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        print(f"\n  Accuracy    : {acc:.4f}")
        print(f"  AUC         : {auc:.4f}")
        print(f"  F1 Score    : {f1:.4f}")

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"\n  Confusion Matrix:")
            print(f"              Pred-0  Pred-1")
            print(f"  Actual 0 :  {tn:6d}  {fp:6d}   (TN / FP)")
            print(f"  Actual 1 :  {fn:6d}  {tp:6d}   (FN / TP)")
            print(f"\n  Sensitivity : {sens:.4f}")
            print(f"  Specificity : {spec:.4f}")
            print(f"  Precision   : {ppv:.4f}")

        print(f"\n{classification_report(all_labels, all_preds, target_names=['negative','positive'])}")


# ── Single patient inference ──────────────────────────────────────

@torch.no_grad()
def predict_single(model, device, patient_dir, fixed_size, threshold):
    img_path = os.path.join(patient_dir, "image.npy")
    if not os.path.exists(img_path):
        print(f"  ERROR: image.npy not found in {patient_dir}")
        return

    image = np.load(img_path, allow_pickle=False).astype(np.float32)

    resize_tf = Compose([
        SpatialPadd(keys=["image"], spatial_size=fixed_size,
                    mode="constant", constant_values=0),
        CenterSpatialCropd(keys=["image"], roi_size=fixed_size),
    ])
    data  = resize_tf({"image": image})
    img_t = _to_plain_tensor(data["image"]).unsqueeze(0).to(device)

    model.eval()
    with autocast(device_type=device.type,
                  enabled=(device.type == "cuda")):
        logit = model(img_t)

    prob   = float(torch.sigmoid(logit).cpu().item())
    result = ("POSITIVE — tumor detected"
              if prob >= threshold else "NEGATIVE — no tumor")

    print(f"\n  Patient     : {os.path.basename(patient_dir)}")
    print(f"  Probability : {prob:.4f}")
    print(f"  Prediction  : {result}  (threshold={threshold})")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_dir", default=None)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--model_path",  default=MODEL_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  Model  : {args.model_path}")

    model = build_model()
    ckpt  = torch.load(args.model_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    fixed_size = tuple(ckpt.get("cfg", {}).get("fixed_size", FIXED_SIZE))
    print(f"  Epoch  : {ckpt.get('epoch','?')}  "
          f"best_score={ckpt.get('best_score','?')}")
    print(f"  Fixed size: {fixed_size}")
    print(f"  Threshold : {args.threshold}\n")

    if args.patient_dir:
        predict_single(model, device, args.patient_dir,
                       fixed_size, args.threshold)
    else:
        evaluate_test_set(model, device, DATA_ROOT,
                          fixed_size, args.threshold)


if __name__ == "__main__":
    main()
