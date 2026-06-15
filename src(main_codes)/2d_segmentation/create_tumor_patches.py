import os
import cv2
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 128

INPUT_DIR = "data/Segmentation_dataset"
OUTPUT_DIR = "data/Segmentation_dataset_patches"

SPLITS = ["train","val"]

def extract_patch(image, mask):

    coords = np.where(mask > 0)

    if len(coords[0]) == 0:
        return None,None

    y_min = np.min(coords[0])
    y_max = np.max(coords[0])
    x_min = np.min(coords[1])
    x_max = np.max(coords[1])

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    half = PATCH_SIZE // 2

    y1 = max(0, center_y - half)
    y2 = y1 + PATCH_SIZE

    x1 = max(0, center_x - half)
    x2 = x1 + PATCH_SIZE

    if y2 > image.shape[0]:
        y2 = image.shape[0]
        y1 = y2 - PATCH_SIZE

    if x2 > image.shape[1]:
        x2 = image.shape[1]
        x1 = x2 - PATCH_SIZE

    patch_img = image[y1:y2, x1:x2]
    patch_mask = mask[y1:y2, x1:x2]

    return patch_img, patch_mask


for split in SPLITS:

    img_dir = os.path.join(INPUT_DIR, split, "images")
    mask_dir = os.path.join(INPUT_DIR, split, "masks")

    out_img = os.path.join(OUTPUT_DIR, split, "images")
    out_mask = os.path.join(OUTPUT_DIR, split, "masks")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)

    files = os.listdir(img_dir)

    print(f"\nProcessing {split} set")

    for name in tqdm(files):

        img_path = os.path.join(img_dir, name)

        mask_name = name.replace(".png","_mask.png")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        patch_img, patch_mask = extract_patch(image, mask)

        if patch_img is None:
            continue

        cv2.imwrite(os.path.join(out_img,name), patch_img)

        cv2.imwrite(
            os.path.join(out_mask,mask_name),
            patch_mask
        )

print("\nPatch dataset created successfully.")