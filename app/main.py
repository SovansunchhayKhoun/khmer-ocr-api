from fastapi import FastAPI, Request
from app.api.endpoints.item import item
from app.api.endpoints.tr_ocr import tr_ocr
from app import settings
from time import perf_counter

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = perf_counter()
    response = await call_next(request)
    process_time = perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def read_root():
    return {
        "version": settings.version,
        "messasge": f"Welcome to {settings.app_name}",
    }


app.include_router(item.router, tags=["Item"])
app.include_router(tr_ocr.router, tags=["Tr Ocr"])
