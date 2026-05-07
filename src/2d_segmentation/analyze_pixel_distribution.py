import os
import cv2
import numpy as np
from tqdm import tqdm

MASK_DIR = "data/Segmentation_dataset/train/masks"

masks = os.listdir(MASK_DIR)

tumor_pixels = 0
background_pixels = 0

print("Analyzing masks...")

for mask_file in tqdm(masks):

    mask_path = os.path.join(MASK_DIR,mask_file)

    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    tumor_pixels += np.sum(mask > 0)

    background_pixels += np.sum(mask == 0)

total_pixels = tumor_pixels + background_pixels

print("\nPixel Statistics")

print("Tumor pixels:", tumor_pixels)
print("Background pixels:", background_pixels)

print("Tumor %:", tumor_pixels/total_pixels*100)
print("Background %:", background_pixels/total_pixels*100)