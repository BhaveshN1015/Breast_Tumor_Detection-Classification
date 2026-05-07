import os
import random
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# PATH
# -----------------------------

DATA_DIR = "data/Segmentation_dataset/train"

IMAGE_DIR = os.path.join(DATA_DIR,"images")
MASK_DIR = os.path.join(DATA_DIR,"masks")

NUM_SAMPLES = 12

# -----------------------------
# LOAD FILE LIST
# -----------------------------

images = os.listdir(IMAGE_DIR)

samples = random.sample(images, NUM_SAMPLES)

# -----------------------------
# VISUALIZE
# -----------------------------

plt.figure(figsize=(12,8))

for i,img_name in enumerate(samples):

    img_path = os.path.join(IMAGE_DIR,img_name)

    mask_name = img_name.replace(".png","_mask.png")

    mask_path = os.path.join(MASK_DIR,mask_name)

    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    # overlay mask
    overlay = image.copy()
    overlay[mask>0] = 255

    plt.subplot(3,4,i+1)
    plt.imshow(overlay,cmap="gray")
    plt.title(img_name)
    plt.axis("off")

plt.suptitle("Segmentation Dataset Alignment Check")

plt.show()