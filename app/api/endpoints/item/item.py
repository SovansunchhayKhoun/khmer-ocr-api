from typing import Union


from fastapi import APIRouter

from app.api.endpoints.item.schema import CreateItemDto, ItemResponse


router = APIRouter(prefix="/item")


@router.get("/get/all/items", response_model=ItemResponse)
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@router.get("/get/one/item/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@router.post("/create/one/item")
def update_item(item: CreateItemDto):
    return {"item_name": item}


@router.put("/update/one/item/{item_id}")
def update_item(item_id: int, item: CreateItemDto):
    return {"item_name": item.name, "item_id": item_id}
