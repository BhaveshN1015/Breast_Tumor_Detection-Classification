import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ---- CHANGE THESE PATHS AFTER TRAINING ----
P1_PATH   = r"D:\Breast_Tumor_AI_Project\data\patients\test\1\P1.nii.gz"
GT_PATH   = r"D:\Breast_Tumor_AI_Project\data\patients\test\1\GT.nii.gz"
PRED_PATH = r"D:\Breast_Tumor_AI_Project\outputs\segmentation_3d_predictions\patient1_pred_mask.nii.gz"

# Load files
p1   = nib.load(P1_PATH).get_fdata()
gt   = nib.load(GT_PATH).get_fdata()
pred = nib.load(PRED_PATH).get_fdata()

# Find the slice with most tumor in GT
tumor_per_slice = gt.sum(axis=(0,1))
best_slice = int(np.argmax(tumor_per_slice))

print(f"Best slice to view : Z={best_slice}")
print(f"GT tumor voxels    : {int(gt.sum())}")
print(f"Pred tumor voxels  : {int(pred.sum())}")

# Compute Dice
intersection = np.logical_and(gt > 0, pred > 0).sum()
dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-8)
print(f"Dice score         : {dice:.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Slice Z={best_slice}  |  Dice={dice:.4f}", fontsize=14)

# MRI only
axes[0].imshow(p1[:,:,best_slice].T, cmap="gray", origin="lower")
axes[0].set_title("MRI (P1)")
axes[0].axis("off")

# MRI + GT overlay
axes[1].imshow(p1[:,:,best_slice].T, cmap="gray", origin="lower")
axes[1].imshow(gt[:,:,best_slice].T, cmap="Greens", alpha=0.5, origin="lower")
axes[1].set_title("MRI + GT Mask (green)")
axes[1].axis("off")

# MRI + Prediction overlay
axes[2].imshow(p1[:,:,best_slice].T, cmap="gray", origin="lower")
axes[2].imshow(pred[:,:,best_slice].T, cmap="Reds", alpha=0.5, origin="lower")
axes[2].set_title("MRI + Prediction (red)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/prediction_check.png", dpi=150)
plt.show()
print("Saved to outputs/prediction_check.png")