from fastapi import FastAPI
from app.api.endpoints.item import item
from app.api.endpoints.tr_ocr import tr_ocr
from app import settings


app = FastAPI()


@app.get("/")
async def read_root():
    return {
        "version": settings.version,
        "messasge": f"Welcome to {settings.app_name}",
    }


app.include_router(item.router, tags=["Item"])
app.include_router(tr_ocr.router, tags=["Tr Ocr"])
