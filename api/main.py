from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import os
import logging
from ml_core import predict_talc_real, process_mask_and_overlay, logger

app = FastAPI(title="Ore Analysis API")

WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    temp_path = os.path.join(WORKSPACE_DIR, f"{task_id}_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Starting analysis for task {task_id}")
        mask, enhanced_img = predict_talc_real(temp_path)

        result_data = process_mask_and_overlay(temp_path, mask, enhanced_img)
        result_data["result_image_path"] = os.path.basename(result_data["result_image_path"])
        result_data["task_id"] = task_id

        return result_data

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))