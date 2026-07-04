import os
import numpy as np
import cv2

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


class ImageLoader:
    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Неподдерживаемый формат: {ext}")

        if ext in (".tif", ".tiff"):
            image = self._load_tiff(path)
        else:
            image = self._load_standard(path)

        return self._normalize_to_uint8_rgb(image)

    def _load_tiff(self, path: str) -> np.ndarray:
        if TIFFFILE_AVAILABLE:
            image = tifffile.imread(path)
            if image.ndim == 4:
                image = image[0]
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[0] < image.shape[-1]:
                image = np.transpose(image, (1, 2, 0))
            return image
        else:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise IOError(f"Не удалось прочитать TIFF: {path}")
            return image

    def _load_standard(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Не удалось прочитать изображение: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _normalize_to_uint8_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image -= image.min()
            max_val = image.max()
            if max_val > 0:
                image = image / max_val * 255.0
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]

        return image

    def get_metadata(self, path: str) -> dict:
        ext = os.path.splitext(path)[1].lower()
        image = self.load(path)
        return {
            "filename": os.path.basename(path),
            "format": ext,
            "shape": image.shape,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        }