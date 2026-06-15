import os
import cv2
import torch
import numpy as np
import random

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, mode="train"):

        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.mode      = mode

        all_images = sorted(os.listdir(image_dir))

        self.tumor_images     = []
        self.non_tumor_images = []

        for img_name in all_images:

            mask_name = img_name.replace(".png", "_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            if mask.sum() > 0:
                self.tumor_images.append(img_name)
            else:
                self.non_tumor_images.append(img_name)

        # ------------------------------------------------------------------
        # TRAIN : all tumor slices + 20% non-tumor  — fixed seed
        # VAL   : all tumor slices + 30% non-tumor  — fixed seed
        # TEST  : ALL slices — completely unfiltered
        # ------------------------------------------------------------------
        if mode == "train":
            rng     = random.Random(42)
            n_keep  = int(len(self.tumor_images) * 0.20)
            sampled = rng.sample(
                self.non_tumor_images,
                min(n_keep, len(self.non_tumor_images))
            )
            self.active_images = self.tumor_images + sampled
            rng.shuffle(self.active_images)

        elif mode == "val":
            rng     = random.Random(42)
            n_keep  = int(len(self.tumor_images) * 0.30)
            sampled = rng.sample(
                self.non_tumor_images,
                min(n_keep, len(self.non_tumor_images))
            )
            self.active_images = self.tumor_images + sampled
            rng.shuffle(self.active_images)

        else:
            # TEST — no sampling, keep every single slice
            self.active_images = self.tumor_images + self.non_tumor_images

        print(f"[{mode}] Tumor slices    : {len(self.tumor_images)}")
        print(f"[{mode}] Non-tumor slices: {len(self.non_tumor_images)}")
        print(f"[{mode}] Active dataset  : {len(self.active_images)}")


    def __len__(self):
        return len(self.active_images)


    def __getitem__(self, index):

        img_name  = self.active_images[index]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 3-CHANNEL: load as BGR color image → shape (H, W, 3)
        # Channels: [0]=P1(B), [1]=P2(G), [2]=P3(R)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Could not load: {img_name}")

        # Resize — already 256x256 from preparation but resize is safe
        image = cv2.resize(image, (256, 256))
        mask  = cv2.resize(mask,  (256, 256))

        # Augmentation BEFORE normalization while still uint8
        # 3-channel albumentations works the same as single-channel
        if self.transform is not None and self.mode == "train":
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        mask  = (mask > 0).astype(np.float32)

        # Convert HWC → CHW for PyTorch: (256,256,3) → (3,256,256)
        image = np.transpose(image, (2, 0, 1))

        # Mask channel dim: (256,256) → (1,256,256)
        mask = np.expand_dims(mask, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask  = torch.tensor(mask,  dtype=torch.float32)

        return image, mask