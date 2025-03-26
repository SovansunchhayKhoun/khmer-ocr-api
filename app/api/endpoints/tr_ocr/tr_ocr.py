import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, HTTPException
import cv2
from PIL import Image
import numpy as np
from app.ml_model import processor, yolo_model, trocr_model
from app.api.endpoints.tr_ocr.schema import (
    FileMetadata,
    Prediction,
    TrOcrPredictionResponseModel,
)
import torch

router = APIRouter(prefix="/tr-ocr")


async def process_file(file: UploadFile) -> TrOcrPredictionResponseModel:
    """Processes a single file and returns a TrOcrPredictionResponseModel."""
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        results = yolo_model(image)

        predictions = []
        cropped_images_list = []
        bounding_boxes = []

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            for box in boxes:
                x, y, width, height = map(int, box)
                cropped_image = image[y:height, x:width]
                cropped_images_list.append(
                    Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                )
                bounding_boxes.append(
                    {
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "width": float(box[2]),
                        "height": float(box[3]),
                    }
                )

        # Batched TrOCR Inference
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
                predictions.append(Prediction(text=text, boundingBox=bounding_boxes[i]))
        print(bounding_boxes)
        file_metadata = FileMetadata(
            fileName=file.filename, fileSize=file.size, fileType=file.content_type
        )

        return TrOcrPredictionResponseModel(
            fileMetadata=file_metadata, predictions=predictions
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=List[TrOcrPredictionResponseModel])
async def predict(files: List[UploadFile]):
    try:
        tasks = [process_file(file) for file in files]
        all_results = await asyncio.gather(*tasks)
        return all_results

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
