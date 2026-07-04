import cv2
import os
import torch
import numpy as np
import logging
import subprocess
import platform
from src.segmentation import TalcPredictor
from src.preprocessing import ImageLoader, PreprocessingPipeline


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("InferencePipeline")


def run_inference(image_path, weights_path=None):
    logger = setup_logger()

    if not os.path.exists(image_path):
        logger.error(f"Image not found at: {image_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"System ready. Running on: {device.upper()}")

    loader = ImageLoader()
    original_img_rgb = loader.load(image_path)
    logger.info(f"Original image loaded. Resolution: {original_img_rgb.shape}")

    predictor = TalcPredictor(
        model_weights_path=weights_path if (weights_path and os.path.exists(weights_path)) else None,
        device=device
    )

    predictor.pipeline = PreprocessingPipeline(
        tile_size=512,
        overlap=64
    )

    logger.info("Starting batch tile analysis")

    mask, enhanced_image_rgb = predictor.predict(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if mask.shape[:2] != original_img_rgb.shape[:2]:
        logger.info("Resizing mask to match original raw resolution...")
        mask_full = cv2.resize(mask, (original_img_rgb.shape[1], original_img_rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = mask

    final_visual = original_img_rgb.copy()
    blue_color = np.array([0, 0, 255], dtype='uint8')
    talc_indices = mask_full > 0.5

    if np.any(talc_indices):
        original_pixels = original_img_rgb[talc_indices]
        blue_fill = np.full_like(original_pixels, blue_color)
        final_visual[talc_indices] = cv2.addWeighted(original_pixels, 0.5, blue_fill, 0.5, 0).reshape(-1, 3)

    step1_path = f"Original_{base_name}.jpg"
    cv2.imwrite(step1_path, cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))

    step2_path = f"Enhanced_{base_name}.jpg"
    cv2.imwrite(step2_path, cv2.cvtColor(enhanced_image_rgb, cv2.COLOR_RGB2BGR))

    step3_path = f"Final_Result_{base_name}.jpg"
    cv2.imwrite(step3_path, cv2.cvtColor(final_visual, cv2.COLOR_RGB2BGR))

    talc_area_pixels = np.sum(mask_full > 0.5)
    talc_percent = (talc_area_pixels / mask_full.size) * 100

    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Detected Talc concentration: {talc_percent:.2f}%")
    logger.info(f"Results saved to project root.")

    if platform.system() == "Darwin":
        subprocess.run(["open", step1_path, step2_path, step3_path])


if __name__ == "__main__":
    TEST_IMAGE = "/Users/PycharmProjects/matsci-modular-analyzer/data/photos_of_ores_pt1/talc-bearing_ores/2550374-2 10х.JPG"
    MODEL_WEIGHTS = "/Users/PycharmProjects/matsci-modular-analyzer/src/segmentation/models/best_talc_model.pth"

    run_inference(TEST_IMAGE, MODEL_WEIGHTS)