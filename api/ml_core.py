import numpy as np
import logging
import cv2
import os
import sys
import torch
import torchvision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation import TalcPredictor
from src.preprocessing import ImageLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML_Core")

SEG_WEIGHTS = os.path.join(PROJECT_ROOT, "src", "segmentation", "weights", "best_talc_model.pth")
CLF_WEIGHTS = os.path.join(PROJECT_ROOT, "src", "classification", "weights", "classifier4.pth")


class OreClassifier:
    def __init__(self, weights_path, device, logger):
        self.device = device
        self.logger = logger
        self.model = torchvision.models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 3)
        )

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded classifier weights: {weights_path}")
        else:
            self.logger.warning("Classifier weights NOT FOUND!")

        self.model.to(device)
        self.model.eval()

    def predict(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()

        return predicted_class_idx, confidence, probabilities.cpu().numpy()


_seg_model = None
_clf_model = None


def get_models():
    global _seg_model, _clf_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _seg_model is None:
        _seg_model = TalcPredictor(
            model_weights_path=SEG_WEIGHTS if os.path.exists(SEG_WEIGHTS) else None,
            device=device,
            logger=logger
        )
    if _clf_model is None:
        _clf_model = OreClassifier(weights_path=CLF_WEIGHTS, device=device, logger=logger)
    return _seg_model, _clf_model


def smart_classify(talc_percent, clf_idx, clf_confidence):
    class_names = ["Talc-bearing Ore", "Ordinary Ore", "Difficult Ore"]

    if clf_idx == 2 and clf_confidence > 0.85:
        return class_names[2]

    if talc_percent > 10.0:
        return class_names[0]

    return class_names[clf_idx]


def run_full_analysis(image_path: str):
    seg_model, clf_model = get_models()

    mask, enhanced_image = seg_model.predict(image_path, batch_size=4)
    clf_idx, clf_conf, all_probs = clf_model.predict(image_path)

    talc_pixels = np.sum(mask > 0.5)
    talc_pct = round((talc_pixels / mask.size) * 100, 2)

    final_class = smart_classify(talc_pct, clf_idx, clf_conf)

    return {
        "mask": mask,
        "enhanced_image": enhanced_image,
        "talc_percent": talc_pct,
        "ore_class": final_class,
        "clf_confidence": round(clf_conf * 100, 1)
    }


def process_mask_and_overlay(image_path: str, analysis_results: dict) -> dict:
    mask = analysis_results["mask"]

    original_img_rgb = ImageLoader().load(image_path)

    if mask.shape[:2] != original_img_rgb.shape[:2]:
        mask_full = cv2.resize(mask, (original_img_rgb.shape[1], original_img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_full = mask

    original_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    final_visual = original_bgr.copy()
    blue_color = np.array([255, 0, 0], dtype='uint8')

    idx = mask_full > 0.5
    if np.any(idx):
        final_visual[idx] = cv2.addWeighted(original_bgr[idx], 0.6, np.full_like(original_bgr[idx], blue_color), 0.4, 0).reshape(-1, 3)

    result_filename = image_path.replace(".", "_result.")
    cv2.imwrite(result_filename, final_visual)

    return {
        "talc_percent": analysis_results["talc_percent"],
        "ore_class": analysis_results["ore_class"],
        "clf_confidence": analysis_results["clf_confidence"],
        "result_image_path": result_filename
    }