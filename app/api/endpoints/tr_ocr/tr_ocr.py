import asyncio
import json
from typing import List
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import numpy as np
from app.api.endpoints.tr_ocr.tr_ocr_service import (
    draw_boxes_on_image,
    draw_predictions_on_image,
    process_and_visualize_file,
    process_file,
)
from app.api.endpoints.tr_ocr.schema import (
    BoundingBox,
    Prediction,
    TrOcrPredictionResponseModel,
)

router = APIRouter(prefix="/tr-ocr")


@router.post("/predict", response_model=List[TrOcrPredictionResponseModel])
async def predict(files: List[UploadFile]):
    try:
        tasks = [process_file(file) for file in files]
        all_results = await asyncio.gather(*tasks)
        return all_results

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/visualize",
    responses={200: {"content": {"image/png": {}}}},
)
async def visualize_with_bbox(file: UploadFile):
    """Endpoint to process the file and return the image with bounding boxes."""
    try:
        prediction_response = await process_and_visualize_file(file)
        image_with_boxes = await draw_boxes_on_image(
            prediction_response.image_bytes, prediction_response.predictions
        )
        return StreamingResponse(io.BytesIO(image_with_boxes), media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/draw",
    responses={200: {"content": {"image/png": {}}}},
)
async def draw_on_image(
    file: UploadFile, text: str, x: str, y: str, width: str, height: str
):
    prediction = Prediction(
        boundingBox=BoundingBox(
            x=float(x), height=float(height), width=float(width), y=float(y)
        ),
        text=text,
    )
    image_bytes = await file.read()
    image_with_boxes = await draw_predictions_on_image(image_bytes, prediction)
    return StreamingResponse(io.BytesIO(image_with_boxes), media_type="image/png")
