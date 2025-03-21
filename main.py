from typing import Union

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
    qty: int


class ItemResponse(BaseModel):
    item_id: int
    q: Union[str, None]


class Message(BaseModel):
    message: str


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get(
    "/get/all/items", response_model=ItemResponse, responses={404: {"model": Message}}
)
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/get/one/item/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/create/one/item")
def update_item(item: Item):
    return {"item_name": item}


@app.put("/update/one/item/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
