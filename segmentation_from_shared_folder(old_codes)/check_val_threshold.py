# save as check_val_threshold.py
import torch, numpy as np, sys, os
sys.path.append('segmentation_from_shared_folder')
from data_3d import build_loaders, PATCH_SIZE
from model_3d import get_model
from monai.inferers import sliding_window_inference

device = torch.device('cuda')
model = get_model(device)
model.load_state_dict(torch.load('models/segmentation_3d/unet3d_best_raw.pth',
                                  map_location=device))
model.eval()

_, val_loader, _ = build_loaders('data/patients_preprocessed', batch_size=1, num_workers=0)

with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        labels = batch['label']
        gt_np = (labels > 0).squeeze().numpy().astype(bool)
        if gt_np.sum() == 0:
            continue

        logits = sliding_window_inference(images, PATCH_SIZE, 4, model, overlap=0.5)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        print(f"Prob map — min:{prob.min():.4f} max:{prob.max():.4f} mean:{prob.mean():.4f}")
        print(f"GT tumor voxels: {gt_np.sum()}")

        for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            pred = (prob > t).astype(bool)
            inter = np.logical_and(pred, gt_np).sum()
            dice = 2*inter / (pred.sum() + gt_np.sum() + 1e-8)
            print(f"  thresh={t:.2f}  pred_voxels={pred.sum():6d}  dice={dice:.4f}")
        print()
        break   # just first tumor patient