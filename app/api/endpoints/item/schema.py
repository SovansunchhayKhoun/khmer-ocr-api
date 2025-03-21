from typing import Union
from pydantic import BaseModel


class ItemResponse(BaseModel):
    item_id: int
    q: Union[str, None]
from typing import Union
from pydantic import BaseModel


class CreateItemDto(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None
    qty: int
