from typing import Union
from fastapi import FastAPI, Body
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

class User(BaseModel):
    username: str
    full_name: Union[str, None] = None

# Mix Path, Query and body parameters
# @app.put('/itmes/{item_id}')
# async def update_item(
#     *, 
#     item_id: int = Path(title='The ID of the item to get', ge=0, le=1000), 
#     q: Union[str, None] = None,
#     item: Union[Item, None] = None):
#     results = {'item_id': item_id}
#     if q:
#         results.update({'q': q})  # type: ignore
#     if item:
#         results.update({'item': item})  # type: ignore
#     return results

# Multiple body parameters
# @app.put('/itmes/{item_id}')
# async def update_item(item_id: int, item: Item, user: User):
#     results = {'item_id': item_id, 'item': item, 'user': user}
#     return results

# Multiple body params and query
# @app.put('/itmes/{item_id}')
# async def update_item( *,
#     item_id: int,
#     item: Item,
#     user: User,
#     importance: int = Body(gt=0),  # gt: 초과
#     q: Union[str, None] = None):
#     results = {'item_id': item_id, 'item': item, 'user': user, 'importance': importance}
#     if q:
#         results.update({'q': q})
#     return results

# Embed a single body parameter
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item = Body(embed=True)):
    results = {"item_id": item_id, "item": item}
    return results