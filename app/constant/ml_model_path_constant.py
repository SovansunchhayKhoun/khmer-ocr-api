from enum import Enum


class MLModelPath(Enum):
    YOLO_V8 = "app/models/ml_models/yolo_v8_173.pt"
    TROCR_KHMER = "app/models/ml_models/fine_tuned_trocr_khmer"
