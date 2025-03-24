import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, HTTPException
import cv2
from PIL import Image
import numpy as np
from app.ml_model import processor, yolo_model, trocr_model
from app.api.endpoints.tr_ocr.schema import TrOcrPredictionResponseModel
import torch

router = APIRouter(prefix="/tr-ocr")


def process_image(image_bytes: bytes) -> List[TrOcrPredictionResponseModel]:
    """Processes a single image."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = yolo_model(image)

    response = []
    cropped_images_list = []
    bounding_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            cropped_images_list.append(
                Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            )
            bounding_boxes.append((box[0], box[1], box[2], box[3]))

    if cropped_images_list:
        pixel_values = processor(
            images=cropped_images_list, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for i, text in enumerate(generated_texts):
            response.append({"text": text, "box": bounding_boxes[i]})

    return response


async def process_file_async(file: UploadFile) -> List[TrOcrPredictionResponseModel]:
    """Asynchronously processes a single file."""
    image_bytes = await file.read()
    return process_image(image_bytes)


@router.post("/predict", response_model=List[List[TrOcrPredictionResponseModel]])
async def predict(files: List[UploadFile]):
    try:
        tasks = [process_file_async(file) for file in files]
        all_results = await asyncio.gather(*tasks)
        return all_results

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
