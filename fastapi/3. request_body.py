from typing import Union
from unittest import result
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Item(BaseModel):  
    name: str  # required
    description: Union[str, None] = None  # Optional
    price: float  # required
    tax: Union[float, None] = None  # Optional

@app.post('/items/')
async def create_item(item: Item):  # 만든 모델로 해당 타입을 선언한다.
    item_dict = item.dict() 
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({'price_with_tax': price_with_tax})
    return item_dict

# Request body + path parameters
# @app.put('/items/{item_id}')
# async def create_item2(item_id: int, item: Item):
#     return {'item_id': item_id, **item.dict()}

# Request body + path + query parameters
@app.put('/items/{item_id}')
async def create_item3(item_id: int, item: Item, q: Union[str, None] = None):
    result = {'item_id': item_id, **item.dict()}
    if q:
        result.update({'q': q})
    return result