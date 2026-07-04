import os
import sys
import logging
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import ImageLoader, ImageEnhancer, AugmentationPipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PreprocessingDemo")

def get_args():
    parser = argparse.ArgumentParser(description="Full Preprocessing Pipeline Demo")
    parser.add_argument(
        "--image",
        type=str,
        default="PycharmProjects/matsci-modular-analyzer/data/photos_of_ores_pt1/talc-bearing_ores/2550374-2 10х.JPG",
        help="Path to the input image"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Whether to apply augmentation"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target size for processing"
    )
    return parser.parse_args()

def run_full_demo():
    args = get_args()

    if not os.path.exists(args.image):
        logger.error(f"Input file not found: {args.image}")
        return

    loader = ImageLoader()
    original = loader.load(args.image)
    logger.info(f"Loaded image with shape: {original.shape}")

    enhancer = ImageEnhancer(normalize_illumination=True, enhance_contrast=True)
    enhanced = enhancer.process(original)
    logger.info("Image enhancement complete")

    final_result = enhanced
    aug_title = ""

    if args.augment:
        aug_engine = AugmentationPipeline(image_size=args.size)
        train_pipe = aug_engine.get_train_pipeline()

        image_for_aug = cv2.resize(enhanced, (args.size, args.size))
        dummy_mask = np.zeros((args.size, args.size), dtype=np.uint8)

        augmented = train_pipe(image=image_for_aug, mask=dummy_mask)
        aug_data = augmented['image']

        if torch.is_tensor(aug_data):
            aug_img = aug_data.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            final_result = np.clip((aug_img * std + mean) * 255, 0, 255).astype(np.uint8)
        else:
            final_result = aug_data.astype(np.uint8)

        aug_title = " + Augmentation"
        logger.info("Augmentation applied")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(original)
    axes[0].set_title("1. ORIGINAL", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(final_result)
    axes[1].set_title(f"2. PROCESSED{aug_title}", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    logger.info("Displaying results")
    plt.show()

if __name__ == "__main__":
    run_full_demo()