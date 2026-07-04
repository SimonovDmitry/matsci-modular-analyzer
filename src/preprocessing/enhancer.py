import cv2
import numpy as np

try:
    from .image_preprocessing import normalize_illumination, denoise
except ImportError:
    from image_preprocessing import normalize_illumination, denoise


class IlluminationNormalizer:
    def apply(self, image_rgb):
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        res_bgr = normalize_illumination(img_bgr, clip_limit=5.0, tile_grid_size=(16, 16))
        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)


class Denoiser:
    def apply(self, image_rgb):
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        res_bgr = denoise(img_bgr, denoise_method='median', kernel_size=3)
        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)


class ContrastEnhancer:
    def apply(self, image_rgb):
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        a = cv2.addWeighted(a, 0.1, np.full_like(a, 128), 0.9, 0)
        b = cv2.addWeighted(b, 0.1, np.full_like(b, 128), 0.9, 0)

        lab = cv2.merge((l, a, b))
        res_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)


class ImageEnhancer:
    def __init__(self, normalize_illumination=True, denoise=False, enhance_contrast=True):
        self.illum = IlluminationNormalizer()
        self.denoiser = Denoiser()
        self.contrast = ContrastEnhancer()

        self.norm_active = normalize_illumination
        self.denoise_active = denoise
        self.contrast_active = enhance_contrast

    def process(self, image_rgb: np.ndarray) -> np.ndarray:
        res = image_rgb.copy()

        if self.contrast_active:
            res = self.contrast.apply(res)

        if self.norm_active:
            res = self.illum.apply(res)

        if self.denoise_active:
            res = self.denoiser.apply(res)

        img_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
        img_bgr = cv2.filter2D(img_bgr, -1, kernel)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)