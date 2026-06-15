"""
evaluate_stage3.py
==================
Stage 3 — Benign vs Malignant evaluation on the test set.

Upgraded to match Stage 1 evaluate_stage1.py patterns:
  ✓ Updated autocast API (device_type string)
  ✓ Consistent output format
  ✓ Single file — no separate predict step needed
  ✓ Threshold sweep + full classification report
  ✓ Per-patient table sorted by probability
  ✓ Saves evaluation CSV

WHY NO SEPARATE PREDICT FILE:
  The predict_evaluate_stage3.py split predict() and evaluate() into
  two steps with an intermediate JSON. This is unnecessary — the model
  is fast on a 64³ patch and runs the entire test set in seconds.
  evaluate_stage3.py runs everything in one pass, exactly like
  evaluate_stage1.py does for Stage 1.

Usage:
    # Full test set evaluation:
    python src/classification/stage3/evaluate_stage3.py

    # Single patient inference:
    python src/classification/stage3/evaluate_stage3.py \
        --patient_dir "D:\\...\\classification_stage3\\test\\malignant\\1"

    # Adjust threshold:
    python src/classification/stage3/evaluate_stage3.py --threshold 0.4
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_stage3  import Stage3Dataset, STAGE3_ROOT, CROP_SIZE, _to_plain_tensor, _crop_patch
from model_stage3 import get_model_stage3

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        accuracy_score, confusion_matrix, classification_report)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── Defaults ──────────────────────────────────────────────────────
DATA_ROOT  = STAGE3_ROOT
MODEL_PATH = (r"D:\Breast_Tumor_AI_Project\models"
              r"\classification_stage3\best_model.pth")
RESULTS_DIR = r"D:\Breast_Tumor_AI_Project\results\classification_stage3"


# ── Full test set evaluation ──────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(model, device, data_root, crop_size, threshold):
    test_ds = Stage3Dataset(data_root, "test", augment=False,
                             crop_size=crop_size)
    loader  = DataLoader(test_ds, batch_size=8, shuffle=False,
                         num_workers=0)

    model.eval()
    all_labels, all_probs, all_preds, all_pids = [], [], [], []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        with autocast(device_type=device.type,
                      enabled=(device.type == "cuda")):
            logits = model(images)

        probs  = torch.sigmoid(logits).cpu().numpy().ravel()
        preds  = (probs >= threshold).astype(int)
        labs   = labels.numpy().ravel().astype(int)

        # Recover patient IDs
        batch_start = i * loader.batch_size
        for j in range(len(probs)):
            idx = batch_start + j
            if idx < len(test_ds.samples):
                pid = os.path.basename(
                    os.path.dirname(test_ds.samples[idx][0]))
                all_pids.append(pid)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labs.tolist())

    print(f"\n  Per-patient results (sorted by probability, high → low):\n")
    print(f"  {'Patient':35s}  {'True':>5s}  {'Prob':>7s}  {'Pred':>5s}  {'OK':>4s}")
    print("  " + "─" * 65)

    sorted_idx = np.argsort(all_probs)[::-1]
    csv_rows   = []
    for idx in sorted_idx:
        t_str = "MAL" if all_labels[idx] == 1 else "BEN"
        p_str = "MAL" if all_preds[idx]  == 1 else "BEN"
        ok    = "✓" if all_labels[idx] == all_preds[idx] else "✗"
        pid   = all_pids[idx] if idx < len(all_pids) else str(idx)
        print(f"  {pid:35s}  {t_str:>5s}  {all_probs[idx]:>7.4f}"
              f"  {p_str:>5s}  {ok:>4s}")
        csv_rows.append({
            "patient_id"     : pid,
            "true_label"     : int(all_labels[idx]),
            "true_class"     : t_str,
            "probability"    : round(float(all_probs[idx]), 4),
            "prediction"     : int(all_preds[idx]),
            "predicted_class": p_str,
            "correct"        : int(all_labels[idx] == all_preds[idx]),
        })

    print("\n" + "─" * 65)

    if SKLEARN_OK and len(set(all_labels)) > 1:
        all_labels_arr = np.array(all_labels)
        all_probs_arr  = np.array(all_probs)
        all_preds_arr  = np.array(all_preds)

        # Threshold sweep
        print("\n  Threshold sweep:")
        print(f"  {'Thresh':>8s}  {'Acc':>7s}  {'AUC':>7s}  "
              f"{'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'Spec':>7s}")
        print("  " + "-" * 60)
        best_f1, best_t = 0.0, threshold
        for t in np.arange(0.2, 0.81, 0.05):
            pp = (all_probs_arr >= t).astype(int)
            acc  = accuracy_score(all_labels_arr, pp)
            auc  = roc_auc_score(all_labels_arr, all_probs_arr)
            f1   = f1_score(all_labels_arr, pp, zero_division=0)
            prec = precision_score(all_labels_arr, pp, zero_division=0)
            rec  = recall_score(all_labels_arr, pp, zero_division=0)
            cm   = confusion_matrix(all_labels_arr, pp, labels=[0,1])
            tn, fp, fn, tp_ = cm.ravel() if cm.size == 4 else (0,0,0,0)
            spec = tn/(tn+fp) if (tn+fp) > 0 else 0
            mark = " ←" if abs(t - threshold) < 0.01 else ""
            print(f"  {t:>8.2f}  {acc:>7.4f}  {auc:>7.4f}  "
                  f"{f1:>7.4f}  {prec:>7.4f}  {rec:>7.4f}  {spec:>7.4f}{mark}")
            if f1 > best_f1:
                best_f1, best_t = f1, round(t, 2)
        print(f"\n  Best F1 threshold: {best_t:.2f}  (F1={best_f1:.4f})")

        # Final at training threshold
        auc = roc_auc_score(all_labels_arr, all_probs_arr)
        cm  = confusion_matrix(all_labels_arr, all_preds_arr, labels=[0,1])
        tn, fp, fn, tp_ = cm.ravel() if cm.size == 4 else (0,0,0,0)
        n_correct = int(np.sum(all_labels_arr == all_preds_arr))

        print(f"\n  ─── Final metrics at threshold={threshold} ───")
        print(f"  Accuracy    : {n_correct}/{len(all_labels)} = "
              f"{n_correct/len(all_labels):.4f}")
        print(f"  AUC-ROC     : {auc:.4f}")
        print(f"  F1 Score    : {f1_score(all_labels_arr,all_preds_arr,zero_division=0):.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"              Pred-BEN  Pred-MAL")
        print(f"  Actual BEN: {tn:8d}  {fp:8d}   (TN / FP)")
        print(f"  Actual MAL: {fn:8d}  {tp_:8d}   (FN / TP)")
        sens = tp_/(tp_+fn) if (tp_+fn) > 0 else 0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0
        ppv  = tp_/(tp_+fp) if (tp_+fp) > 0 else 0
        print(f"\n  Sensitivity : {sens:.4f}")
        print(f"  Specificity : {spec:.4f}")
        print(f"  Precision   : {ppv:.4f}")
        print(f"\n{classification_report(all_labels_arr, all_preds_arr, target_names=['Benign','Malignant'])}")

    # Save CSV
    if csv_rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, "stage3_evaluation.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV saved → {csv_path}")


# ── Single patient inference ──────────────────────────────────────

@torch.no_grad()
def predict_single(model, device, patient_dir, crop_size, threshold):
    img_path = os.path.join(patient_dir, "image.npy")
    lbl_path = os.path.join(patient_dir, "label.npy")
    if not os.path.exists(img_path):
        print(f"  ERROR: image.npy not found in {patient_dir}")
        return

    img = np.load(img_path, allow_pickle=False).astype(np.float32)
    lbl = (np.load(lbl_path, allow_pickle=False).astype(np.float32)
           if os.path.exists(lbl_path) else None)

    patch = _crop_patch(img, lbl, crop_size)
    patch = np.clip(patch, -3.5, 7.0)
    img_t = _to_plain_tensor(patch).unsqueeze(0).to(device)

    model.eval()
    with autocast(device_type=device.type,
                  enabled=(device.type == "cuda")):
        logit = model(img_t)

    prob   = float(torch.sigmoid(logit).cpu().item())
    result = ("MALIGNANT" if prob >= threshold else "BENIGN")
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

    model = get_model_stage3(device)
    ckpt  = torch.load(args.model_path, map_location=device,
                       weights_only=False)
    # Support both state_dict-only saves and full checkpoint saves
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    crop_size   = ckpt.get("cfg", {}).get("crop_size", CROP_SIZE) \
                  if isinstance(ckpt, dict) else CROP_SIZE
    best_score  = ckpt.get("best_score", "?") if isinstance(ckpt, dict) else "?"
    epoch       = ckpt.get("epoch", "?")      if isinstance(ckpt, dict) else "?"
    print(f"  Checkpoint : epoch={epoch}  best_score={best_score}")
    print(f"  Crop size  : {crop_size}³")
    print(f"  Threshold  : {args.threshold}\n")

    if args.patient_dir:
        predict_single(model, device, args.patient_dir,
                       crop_size, args.threshold)
    else:
        evaluate_test_set(model, device, DATA_ROOT,
                          crop_size, args.threshold)


if __name__ == "__main__":
    main()
