from pydantic import BaseModel


class TrOcrPredictionResponseModel(BaseModel):
    text: str
    box: list[float]
