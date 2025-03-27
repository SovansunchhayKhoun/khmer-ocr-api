from typing import List
from pydantic import BaseModel


class FileMetadata(BaseModel):
    fileName: str
    fileSize: int
    fileType: str


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Prediction(BaseModel):
    text: str
    boundingBox: BoundingBox


class TrOcrPredictionResponseModel(BaseModel):
    fileMetadata: FileMetadata
    predictions: List[Prediction]


class TrOcrImagePredictionResponseModel(BaseModel):
    fileMetadata: FileMetadata
    predictions: List[Prediction]
    image_bytes: bytes
