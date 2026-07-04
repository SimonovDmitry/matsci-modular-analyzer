import logging

from .image_loader import ImageLoader
from .color_normalizer import ColorNormalizer, GrayscaleFallback
from .enhancer import ImageEnhancer
from .tiles_utils import TileProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    def __init__(self, tile_size: int = 512, overlap: int = 64, normalize_illumination: bool = True,
        denoise: bool = False, normalize_color: bool = True, use_grayscale_fallback: bool = False):
        self.loader = ImageLoader()
        self.enhancer = ImageEnhancer(
            normalize_illumination=normalize_illumination,
            denoise=denoise,
        )
        self.tile_size = tile_size
        self.overlap = overlap

        self.normalize_color = normalize_color
        self.use_grayscale_fallback = use_grayscale_fallback
        self.color_normalizer = None
        self.grayscale_fallback = None
        self._color_normalization_active = False

        if use_grayscale_fallback:
            self.grayscale_fallback = GrayscaleFallback()
            self._color_normalization_active = True
        elif normalize_color:
            self.color_normalizer = ColorNormalizer()

        self.config = {
            "tile_size": tile_size,
            "overlap": overlap,
            "normalize_illumination": normalize_illumination,
            "denoise": denoise,
            "normalize_color": normalize_color,
            "use_grayscale_fallback": use_grayscale_fallback,
            "color_normalization_active": self._color_normalization_active,
        }

    def run(self, image_path: str):
        logger.info(f"Обработка: {image_path}")

        image = self.loader.load(image_path)
        logger.info(f"Загружено изображение размером {image.shape}")

        color_normalized = image
        if self.use_grayscale_fallback:
            color_normalized = self.grayscale_fallback.apply(image)
            logger.info("Применён grayscale fallback (цвет полностью убран)")
        elif self._color_normalization_active and self.color_normalizer is not None:
            color_normalized = self.color_normalizer.apply(image)
            logger.info("Применена Reinhard цветовая нормализация относительно эталона")

        enhanced = self.enhancer.process(color_normalized)

        tile_processor = TileProcessor(tile_size=self.tile_size, overlap=self.overlap)
        tiles, coords = tile_processor.split_into_tiles(enhanced)
        logger.info(f"Разбито на {len(tiles)} тайлов")

        return tile_processor, tiles, coords, enhanced

    def get_config(self) -> dict:
        return dict(self.config)