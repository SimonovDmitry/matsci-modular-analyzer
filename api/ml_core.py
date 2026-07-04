import numpy as np
import logging
import time
import random
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def predict_ore_class_mock(image_path: str) -> str:
    """
    Заглушка Нейросети №1 (Классификация).
    Теперь она САМА определяет класс руды, без нашей бизнес-логики.
    """
    logger.info(f"[NN1] Старт классификации руды для: {image_path}")
    time.sleep(1)  # Имитация работы

    # Модель сразу выдает один из трех вариантов
    predicted_class = random.choice(["Оталькованная руда", "Рядовая руда", "Труднообогатимая руда"])

    logger.info(f"[NN1] Классификация завершена: {predicted_class}")
    return predicted_class


def predict_talc_probabilities_mock(image_path: str) -> np.ndarray:
    """
    Заглушка Нейросети №2 (Сегментация).
    Возвращает матрицу вероятностей от 0.0 до 1.0.
    """
    logger.info(f"[NN2] Старт генерации матрицы вероятностей для: {image_path}")
    time.sleep(2)  # Имитация работы

    height, width = 512, 512
    # Генерируем матрицу случайных вероятностей
    prob_matrix = np.random.uniform(low=0.0, high=1.0, size=(height, width))

    # Искусственно занижаем вероятности, чтобы талька не было слишком много на заглушке
    prob_matrix = prob_matrix * 0.3

    logger.info("[NN2] Матрица вероятностей сгенерирована")
    return prob_matrix


def process_mask_and_overlay(image_path: str, prob_matrix: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Переводит вероятности в бинарную маску, считает процент площади
    и сохраняет картинку с наложенным цветом.
    """
    logger.info(f"Применение порога {threshold} и отрисовка маски...")

    binary_mask = prob_matrix >= threshold

    total_pixels = binary_mask.size
    talc_pixels = np.sum(binary_mask)
    talc_pct = (talc_pixels / total_pixels) * 100

    image = cv2.imread(image_path)

    if image is None:
        image = np.zeros((prob_matrix.shape[0], prob_matrix.shape[1], 3), dtype=np.uint8)

    image = cv2.resize(image, (prob_matrix.shape[1], prob_matrix.shape[0]))

    color_overlay = np.zeros_like(image)
    color_overlay[binary_mask] = [255, 0, 0]

    alpha = 0.3
    result_image = cv2.addWeighted(color_overlay, alpha, image, 1 - alpha, 0)

    result_filename = image_path.replace(".", "_result.")
    cv2.imwrite(result_filename, result_image)

    logger.info(f"Изображение с маской сохранено по пути: {result_filename}")

    return {
        "talc_percent": round(talc_pct, 2),
        "result_image_path": result_filename
    }