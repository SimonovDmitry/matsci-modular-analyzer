import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# Добавляем корень проекта в пути поиска, чтобы импорты src работали
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import (
    ImageLoader,
    ColorNormalizer,
    ImageEnhancer,
    AugmentationPipeline
)

# ==========================================
# НАСТРОЙКИ ДЛЯ ЗАПУСКА ИЗ IDE (БЕЗ КОНСОЛИ)
# ==========================================
# 1. Пропиши здесь путь к любому своему фото
IMAGE_PATH = "/Users/yahikonetalant/PycharmProjects/matsci-modular-analyzer/data/photos_of_ores_pt1/talc-bearing_ores/2550374-2 10х.JPG"

# 2. Путь к статистике цветов (создастся сам, если нет)
REF_STATS_PATH = "configs/reference_stats.json"

# 3. Нужно ли применять аугментацию (поворот, шум, деформацию)?
APPLY_AUGMENTATION = True


# ==========================================

def run_full_demo():
    print("🚀 Запуск полного пайплайна обработки...")

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Файл не найден: {IMAGE_PATH}")
        return

    # 1. ЗАГРУЗКА
    loader = ImageLoader()
    original = loader.load(IMAGE_PATH)  # RGB
    print(f"✅ Изображение загружено: {original.shape}")

    # 2. ЦВЕТОВАЯ НОРМАЛИЗАЦИЯ (Reinhard)
    normalizer = ColorNormalizer()
    if os.path.exists(REF_STATS_PATH):
        normalizer.load_reference_stats(REF_STATS_PATH)
    else:
        print("💡 Файл статистики не найден, создаю эталон из этого же фото...")
        normalizer.fit_reference(original)
        os.makedirs(os.path.dirname(REF_STATS_PATH), exist_ok=True)
        normalizer.save_reference_stats(REF_STATS_PATH)

    color_fixed = normalizer.apply(original)
    print("✅ Цвета выровнены по эталону.")

    # 3. УЛУЧШЕНИЕ ОСВЕЩЕНИЯ (Твой агрессивный CLAHE + Sharpening)
    enhancer = ImageEnhancer(normalize_illumination=True, enhance_contrast=True)
    enhanced = enhancer.process(color_fixed)
    print("✅ Освещение выровнено, резкость повышена.")

    final_result = enhanced
    aug_title = ""

    # 4. АУГМЕНТАЦИЯ (Твой сложный пайплайн с Dropout и деформациями)
    if APPLY_AUGMENTATION:
        # Создаем "пустую" маску, так как AugmentationPipeline требует пару (img, mask)
        dummy_mask = np.zeros(enhanced.shape[:2], dtype=np.uint8)

        # Берем твой класс-обертку (который мы создали в __init__.py)
        aug_engine = AugmentationPipeline(image_size=512)
        train_pipe = aug_engine.get_train_pipeline()

        # Применяем!
        augmented = train_pipe(image=enhanced, mask=dummy_mask)

        # Конвертируем тензор обратно в numpy для отрисовки
        aug_img = augmented['image'].permute(1, 2, 0).numpy()

        # Де-нормализация (для корректного отображения в matplotlib)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        final_result = np.clip((aug_img * std + mean) * 255, 0, 255).astype(np.uint8)
        aug_title = " + Augmentation"
        print("✅ Применены случайные аугментации (Rotation, Noise, Elastic).")

    # 5. ВИЗУАЛИЗАЦИЯ
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(original)
    axes[0].set_title("1. ОРИГИНАЛ (Как есть)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(final_result)
    axes[1].set_title(f"2. ПОЛНАЯ ОБРАБОТКА{aug_title}", fontsize=12, color="green")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    print("\n✨ Готово! На экране сравнение 'До' и 'После'.")


if __name__ == "__main__":
    run_full_demo()