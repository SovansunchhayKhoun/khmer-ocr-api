from fastapi import FastAPI
from app.api.endpoints.item import item
from app import settings


app = FastAPI()


@app.get("/")
async def read_root():
    return {
        "version": settings.version,
        "messasge": f"Welcome to {settings.app_name}",
    }


app.include_router(item.router, tags=["Item"])
