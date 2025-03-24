from pydantic import BaseModel


# class FileMetadata:
#     fileName: str
#     fileSize: int
#     fileType: str


# class Prediction:
#     text: str
#     box: list[float]


# class TrOcrPredictionResponseModel(BaseModel):
#     fileMetadata: FileMetadata
#     prediction: Prediction


class TrOcrPredictionResponseModel(BaseModel):
    text: str
    box: list[float]
