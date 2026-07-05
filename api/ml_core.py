import numpy as np
import logging
import cv2
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation import TalcPredictor
from src.preprocessing import ImageLoader

logger = logging.getLogger("ML_Core")

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "src", "segmentation", "weights", "best_talc_model.pth")
_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        logger.info(f"Initializing model: {WEIGHTS_PATH}")
        w_path = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else None
        _predictor_instance = TalcPredictor(
            model_weights_path=w_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            logger=logger
        )
    return _predictor_instance


def predict_talc_real(image_path: str):
    predictor = get_predictor()
    # Обработка изображения
    mask, enhanced_image = predictor.predict(image_path, batch_size=1)
    return mask, enhanced_image


def classify_ore_logic(talc_percent: float) -> str:
    if talc_percent > 10.0:
        return "Talc-bearing Ore"
    return "Ordinary Ore"


def process_mask_and_overlay(image_path: str, mask: np.ndarray, enhanced_image: np.ndarray) -> dict:
    loader = ImageLoader()
    original_img_rgb = loader.load(image_path)

    if mask.shape[:2] != original_img_rgb.shape[:2]:
        mask_full = cv2.resize(mask, (original_img_rgb.shape[1], original_img_rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = mask

    talc_pixels = np.sum(mask_full > 0.5)
    talc_pct = round((talc_pixels / mask_full.size) * 100, 2)

    original_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    final_visual = original_bgr.copy()

    blue_color = np.array([255, 0, 0], dtype='uint8')  # Синий в BGR
    talc_indices = mask_full > 0.5

    if np.any(talc_indices):
        original_pixels = original_bgr[talc_indices]
        blue_fill = np.full_like(original_pixels, blue_color)
        final_visual[talc_indices] = cv2.addWeighted(original_pixels, 0.6, blue_fill, 0.4, 0).reshape(-1, 3)

    result_filename = image_path.replace(".", "_result.")
    cv2.imwrite(result_filename, final_visual)

    return {
        "talc_percent": talc_pct,
        "ore_class": classify_ore_logic(talc_pct),
        "result_image_path": result_filename
    }