import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp


MODEL_PATH  = "models/smooth_0.8157/unet_best.pth"   # use raw best — highest single-epoch Dice
TEST_IMAGES = "data/Segmentation_dataset/test/images"
OUTPUT_DIR  = "outputs/segmentation_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD   = 0.1    # optimal threshold from sweep
TTA_ENABLED = True   # Test-Time Augmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# UPDATED: in_channels=3 to match 3-channel training
model = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = None,
    in_channels     = 3,        # 3-channel DCE-MRI (P1+P2+P3)
    classes         = 1,
    activation      = None
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully")


def preprocess(image_np):
    """
    Resize to 256x256, normalize to [0,1].
    Input:  (H, W, 3) uint8 BGR image
    Output: (1, 3, H, W) float32 tensor
    """
    img = cv2.resize(image_np, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))              # (256,256,3) → (3,256,256)
    img = torch.tensor(img).unsqueeze(0).float().to(device)  # (1,3,256,256)
    return img


def predict_single(image_tensor):
    with torch.no_grad():
        out  = model(image_tensor)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    return prob


def predict_with_tta(image_np):
    """
    Test-Time Augmentation on 3-channel images.
    Average predictions from original + 3 augmented versions.
    All flips/rotations applied to the full 3-channel image.
    """
    prob_sum = predict_single(preprocess(image_np))

    # Horizontal flip — flip along width axis
    flipped_h = cv2.flip(image_np, 1)
    p_h = predict_single(preprocess(flipped_h))
    prob_sum += cv2.flip(p_h, 1)

    # Vertical flip — flip along height axis
    flipped_v = cv2.flip(image_np, 0)
    p_v = predict_single(preprocess(flipped_v))
    prob_sum += cv2.flip(p_v, 0)

    # 180 degree rotation
    rotated = cv2.rotate(image_np, cv2.ROTATE_180)
    p_r = predict_single(preprocess(rotated))
    prob_sum += cv2.rotate(p_r, cv2.ROTATE_180)

    return prob_sum / 4.0


def remove_noise_blobs(mask, min_size=10):
    """Remove only pure noise blobs under 10px — keep all genuine tumor regions."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned


def apply_morphological_closing(mask):
    """
    Connects nearby tumor fragments and fills small holes.
    Improves Dice without retraining.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


img_files = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(".png")])
print(f"Running inference on {len(img_files)} images...")

for img_name in img_files:

    img_path = os.path.join(TEST_IMAGES, img_name)

    # UPDATED: load as 3-channel color image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)   # (H, W, 3)

    if image is None:
        print(f"  Warning: could not read {img_name}, skipping.")
        continue

    # UPDATED: use [:2] to ignore channel dimension
    original_h, original_w = image.shape[:2]

    if TTA_ENABLED:
        prob_map = predict_with_tta(image)
    else:
        prob_map = predict_single(preprocess(image))

    binary_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

    binary_mask = remove_noise_blobs(binary_mask, min_size=10)

    binary_mask = apply_morphological_closing(binary_mask)

    binary_mask = cv2.resize(
        binary_mask, (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), binary_mask)

print(f"Predictions saved to: {OUTPUT_DIR}")