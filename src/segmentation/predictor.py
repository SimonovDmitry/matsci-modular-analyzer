import torch
import numpy as np
import os
from src.preprocessing.tiles_utils import TileProcessor
from src.preprocessing.image_loader import ImageLoader
from src.preprocessing.enhancer import ImageEnhancer

try:
    from .model import TalcSegmentationModel
except (ImportError, ValueError):
    from model import TalcSegmentationModel


class TalcPredictor:
    def __init__(self, model_weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TalcSegmentationModel().to(self.device).float()
        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_weights(model_weights_path, device=self.device)
        self.model.eval()

        self.loader = ImageLoader()
        self.enhancer = ImageEnhancer()
        self.tile_processor = TileProcessor(tile_size=512, overlap=64)

    def predict(self, image_path, batch_size=4):
        img = self.loader.load(image_path)
        enhanced = self.enhancer.process(img)

        tiles, coords = self.tile_processor.split_into_tiles(enhanced)

        all_preds = []
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            batch_tensor = self.tile_processor.tiles_to_tensor(batch, device=self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch_tensor))
                all_preds.append(probs.cpu())

        final_probs = torch.cat(all_preds, dim=0)
        mask = self.tile_processor.stitch_tiles(final_probs)
        return (mask > 127).astype(np.uint8), enhanced