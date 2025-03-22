from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from app.config.settings import Settings
from app.constant.ml_model_path_constant import MLModelPath

settings = Settings()

# Load models outside the predict function (load once at startup)
yolo_model = YOLO(MLModelPath.YOLO_V8.value)
processor = TrOCRProcessor.from_pretrained(MLModelPath.TROCR_KHMER.value, use_fast=True)
trocr_model = VisionEncoderDecoderModel.from_pretrained(MLModelPath.TROCR_KHMER.value)
