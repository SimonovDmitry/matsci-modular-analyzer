import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset


class TalcDataset(Dataset):
    _valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.JPG')
    _mask_suffixes = ['_mask', '_mack', '_masks']

    def __init__(self, image_paths, mask_paths, transform=None, logger=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Dataset initialized with {len(self.image_paths)} pairs.")

    @classmethod
    def from_root(cls, data_root, transform=None, logger=None):
        img_paths, mask_paths = [], []
        tmp_logger = logger or logging.getLogger(__name__)

        abs_root = os.path.abspath(data_root)
        tmp_logger.info(f"Scanning root directory: {abs_root}")

        if not os.path.exists(abs_root):
            tmp_logger.error(f"Root directory not found: {abs_root}")
            return cls([], [], transform, logger)

        for root, dirs, _ in os.walk(abs_root):
            for d in dirs:
                if any(d.lower().endswith(s) for s in cls._mask_suffixes):
                    continue

                img_dir = os.path.join(root, d)

                for s in cls._mask_suffixes:
                    mask_dir = os.path.join(root, d + s)
                    if os.path.exists(mask_dir) and os.path.isdir(mask_dir):
                        i, m = cls._match_files(img_dir, mask_dir, tmp_logger)
                        img_paths.extend(i)
                        mask_paths.extend(m)
                        break

        return cls(img_paths, mask_paths, transform, logger)

    @staticmethod
    def _match_files(img_dir, mask_dir, logger):
        imgs, masks = [], []
        mask_files = os.listdir(mask_dir)

        files = [f for f in os.listdir(img_dir) if f.lower().endswith(TalcDataset._valid_ext)]

        for f in files:
            name, ext = os.path.splitext(f)
            found = False

            for s in TalcDataset._mask_suffixes:
                for m_ext in [ext, ext.lower(), ext.upper(), '.png', '.jpg', '.JPG']:
                    target_mask_name = f"{name}{s}{m_ext}"

                    if target_mask_name in mask_files:
                        imgs.append(os.path.join(img_dir, f))
                        masks.append(os.path.join(mask_dir, target_mask_name))
                        found = True
                        break
                if found: break

            if not found:
                logger.warning(f"No mask found for: {f} in {os.path.basename(mask_dir)}")

        return imgs, masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask