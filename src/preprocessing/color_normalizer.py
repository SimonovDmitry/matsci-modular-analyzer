import json
import os
import cv2
import numpy as np


class ColorNormalizer:
    def __init__(self):
        self.reference_mean = None
        self.reference_std = None

    def fit_reference(self, reference_image):
        lab = cv2.cvtColor(reference_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.reference_mean = lab.reshape(-1, 3).mean(axis=0)
        self.reference_std = lab.reshape(-1, 3).std(axis=0)
        self.reference_std[self.reference_std < 1e-6] = 1e-6

    def fit_reference_from_multiple(self, reference_images):
        means, stds = [], []
        for image in reference_images:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            means.append(lab.reshape(-1, 3).mean(axis=0))
            stds.append(lab.reshape(-1, 3).std(axis=0))
        self.reference_mean = np.mean(means, axis=0)
        self.reference_std = np.mean(stds, axis=0)
        self.reference_std[self.reference_std < 1e-6] = 1e-6

    def apply(self, image):
        if self.reference_mean is None or self.reference_std is None:
            raise RuntimeError("Reference statistics not set. Please call fit_reference()"
                "or fit_reference_from_multiple(), or load_reference_stats() first")

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_mean = lab.reshape(-1, 3).mean(axis=0)
        source_std = lab.reshape(-1, 3).std(axis=0)
        source_std[source_std < 1e-6] = 1e-6

        normalized = (lab - source_mean) / source_std * self.reference_std + self.reference_mean
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)

    def save_reference_stats(self, path):
        if self.reference_mean is None:
            raise RuntimeError("Nothing to save call fit_reference() first.")
        stats = {
            "reference_mean": self.reference_mean.tolist(),
            "reference_std": self.reference_std.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def load_reference_stats(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference stats file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.reference_mean = np.array(stats["reference_mean"], dtype=np.float32)
        self.reference_std = np.array(stats["reference_std"], dtype=np.float32)


class GrayscaleFallback:
    def apply(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        return cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)