from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os

from ml_core import (
    predict_ore_class_mock,
    predict_talc_probabilities_mock,
    process_mask_and_overlay,
    logger
)

app = FastAPI(
    title="🔬 Норникель: Анализ микроструктуры руды",
    description="API-сервис для автоматизированной классификации руд и сегментации талька на панорамных OM-изображениях.",
    version="1.0.0"
)

tasks_db = {}
WORKSPACE_DIR = "workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

from ml_core import predict_talc_real, process_mask_and_overlay, logger


def process_image_task(task_id: str, file_path: str):
    try:
        tasks_db[task_id]["status"] = "processing"

        # 1. ЗАПУСК РЕАЛЬНОЙ НЕЙРОСЕТИ
        mask, enhanced_image = predict_talc_real(file_path)

        # 2. ПОСТПРОЦЕССИНГ (Маска + Проценты + Класс)
        mask_data = process_mask_and_overlay(file_path, mask, enhanced_image)

        analysis_result = {
            "ore_class": mask_data["ore_class"],
            "talc_percent": mask_data["talc_percent"],
            "result_image_path": mask_data["result_image_path"]
        }

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = analysis_result
        logger.info(f"Task {task_id} successful.")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)


@app.post(
    "/upload",
    summary="Загрузить изображение шлифа",
    description="Принимает файл изображения (TIFF, PNG, JPEG), сохраняет его на сервере и запускает фоновую задачу анализа двумя нейросетями.",
    tags=["Анализ руды"] # Тег создаст красивую визуальную группу
)
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    file_path = os.path.join(WORKSPACE_DIR, f"{task_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    logger.info(f"Файл {file.filename} сохранен. Идентификатор задачи: {task_id}")

    tasks_db[task_id] = {"status": "queued", "result": None}
    background_tasks.add_task(process_image_task, task_id, file_path)

    return JSONResponse(content={"task_id": task_id, "status": "queued"})


@app.get(
    "/status/{task_id}",
    summary="Проверить статус и получить результат",
    description="Возвращает текущий статус обработки (queued, processing, completed, failed) и результаты анализа (процент талька, класс руды, путь к картинке), если задача завершена.",
    tags=["Мониторинг"]
)
async def get_status(task_id: str):
    if task_id not in tasks_db:
        return JSONResponse(status_code=404, content={"error": "Задача не найдена"})
    return tasks_db[task_id]