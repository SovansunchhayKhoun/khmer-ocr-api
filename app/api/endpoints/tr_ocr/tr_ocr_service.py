from typing import List
from fastapi import UploadFile, HTTPException
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from app.constant.font_path_constant import FontPathConstant
from app.ml_model import processor, yolo_model, trocr_model
from app.api.endpoints.tr_ocr.schema import (
    FileMetadata,
    Prediction,
    TrOcrImagePredictionResponseModel,
    TrOcrPredictionResponseModel,
)
import torch


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

        file_metadata = FileMetadata(
            fileName=file.filename, fileSize=file.size, fileType=file.content_type
        )

        return TrOcrPredictionResponseModel(
            fileMetadata=file_metadata, predictions=predictions
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def process_and_visualize_file(
    file: UploadFile,
) -> TrOcrImagePredictionResponseModel:
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

        file_metadata = FileMetadata(
            fileName=file.filename, fileSize=file.size, fileType=file.content_type
        )

        return TrOcrImagePredictionResponseModel(
            fileMetadata=file_metadata,
            predictions=predictions,
            image_bytes=image_bytes,
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


async def draw_boxes_on_image(
    image_bytes: bytes, predictions: List[Prediction]
) -> bytes:
    """Draws bounding boxes and text on an image based on prediction data."""
    try:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        try:
            font = ImageFont.truetype(FontPathConstant.BATTAMBANG.value, 15)
        except IOError:
            font = ImageFont.load_default()
            print("Warning: Arial font not found, using default font.")

        for prediction in predictions:
            bbox = prediction.boundingBox
            x_min = int(bbox.x)
            y_min = int(bbox.y)
            x_max = int(bbox.width)
            y_max = int(bbox.height)
            text = prediction.text

            # Draw rectangle
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)

            # Draw text above the bounding box
            text_x = x_min
            text_y = y_min - 15  # Adjust vertical position as needed
            draw.text((text_x, text_y), text, fill="red", font=font)

        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    except Exception as e:
        print(e)
        raise e


async def draw_predictions_on_image(
    image_bytes: bytes, prediction: Prediction
) -> bytes:
    """Draws bounding boxes and text on an image based on a list of Prediction objects."""
    try:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        try:
            font = ImageFont.truetype(FontPathConstant.BATTAMBANG.value, 15)
        except IOError:
            font = ImageFont.load_default()
            print("Warning: Battambang font not found, using default font.")

        print(prediction)

        bbox = prediction.boundingBox
        x_min = int(bbox.x)
        y_min = int(bbox.y)
        x_max = int(bbox.width)
        y_max = int(bbox.height)
        text = prediction.text

        # Draw rectangle
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)

        # Draw text above the bounding box
        text_x = x_min
        text_y = y_min - 15  # Adjust vertical position as needed
        draw.text((text_x, text_y), text, fill="red", font=font)

        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
