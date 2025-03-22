from time import perf_counter
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from app.constant.ml_model_path_constant import MLModelPath


def load_models():
    """Load all models and return them."""
    t1_start = perf_counter()
    print("Loading models...")

    # Load YOLO model
    yolo_model = YOLO(MLModelPath.YOLO_V8.value)

    # Load TrOCR processor and model
    processor = TrOCRProcessor.from_pretrained(
        MLModelPath.TROCR_KHMER.value, use_fast=True
    )
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        MLModelPath.TROCR_KHMER.value
    )

    print(f"Finished loading models, took {perf_counter() - t1_start:.2f}s")
    return yolo_model, processor, trocr_model


# Load models at startup
yolo_model, processor, trocr_model = load_models()
