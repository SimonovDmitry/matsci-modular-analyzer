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
    def __init__(self, model_weights_path=None, device=None, logger=None):
        self.logger = logger
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TalcSegmentationModel().to(self.device).float()

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_weights(model_weights_path, device=self.device)
        self.model.eval()

        self.loader = ImageLoader()
        self.enhancer = ImageEnhancer()
        # Создаем процессор, но будем использовать только его логику нарезки
        self.tile_processor = TileProcessor(tile_size=512, overlap=64)

    def predict(self, image_path, batch_size=4):
        img = self.loader.load(image_path)
        enhanced = self.enhancer.process(img)

        # 1. Используем только нарезку из TileProcessor
        tiles, coords = self.tile_processor.split_into_tiles(enhanced)

        all_preds = []
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]

            # 2. РЕАЛИЗУЕМ ПРАВИЛЬНУЮ КОНВЕРТАЦИЮ В ТЕНЗОР ПРЯМО ЗДЕСЬ
            # (Исправляем ошибку negative stride без изменения внешнего файла)
            img_rgb = batch[..., ::-1].copy()  # Добавляем .copy()
            batch_tensor = torch.from_numpy(img_rgb).permute(0, 3, 1, 2).float() / 255.0
            batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                # 3. ИСПРАВЛЯЕМ ОШИБКУ .cpu() ПРЯМО ЗДЕСЬ
                probs = torch.sigmoid(self.model(batch_tensor))
                all_preds.append(probs.detach().cpu().numpy())

        # Собираем все предсказания в один массив
        final_probs = np.concatenate(all_preds, axis=0)

        # 4. РЕАЛИЗУЕМ ПРАВИЛЬНУЮ СКЛЕЙКУ (Копируем логику, но с исправлениями)
        mask = self._fixed_stitch(final_probs)

        return mask, enhanced

    def _fixed_stitch(self, predictions):
        """Исправленная версия склейки без ошибок размерностей."""
        h, w = self.tile_processor.h, self.tile_processor.w
        tile_size = self.tile_processor.tile_size
        overlap = self.tile_processor.overlap

        result = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for idx, (x, y) in enumerate(self.tile_processor.coords):
            pred = predictions[idx]
            if pred.ndim == 3: pred = pred[0]  # Убираем канал (1, 512, 512) -> (512, 512)

            weight = np.ones((tile_size, tile_size), dtype=np.float32)
            if overlap > 0:
                ramp = np.linspace(0, 1, overlap)
                weight[:overlap, :] *= ramp[:, None]
                weight[-overlap:, :] *= ramp[::-1, None]
                weight[:, :overlap] *= ramp[None, :]
                weight[:, -overlap:] *= ramp[None, ::-1]

            result[y:y + tile_size, x:x + tile_size] += pred * weight
            weight_map[y:y + tile_size, x:x + tile_size] += weight

        result /= (weight_map + 1e-8)
        return (result > 0.5).astype(np.uint8)  # Возвращаем сразу 0 или 1