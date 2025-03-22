from fastapi import APIRouter, UploadFile, HTTPException
import cv2
from PIL import Image
import numpy as np
from app import processor, yolo_model, trocr_model

from app.constant.ml_model_path_constant import MLModelPath

router = APIRouter(prefix="/tr-ocr")


@router.post("/predict")
async def predict(file: UploadFile):
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        results = yolo_model(image)
        cropped_images = []
        bounding_boxes = []

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = image[y1:y2, x1:x2]
                cropped_images.append(
                    Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                )
                bounding_boxes.append((x1, y1, x2, y2))

        texts = []
        for cropped_image in cropped_images:
            pixel_values = processor(
                images=cropped_image, return_tensors="pt"
            ).pixel_values
            generated_ids = trocr_model.generate(pixel_values)
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            texts.append(generated_text)

        response = []
        for i in range(len(texts)):
            response.append({"text": texts[i], "box": bounding_boxes[i]})

        return response

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
