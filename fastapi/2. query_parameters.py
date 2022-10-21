from fastapi import FastAPI
from typing import Union

app = FastAPI()

fake_items_db = [{'item_name': 'Foo'}, {'item_name': 'Bar'}, {'item_name': 'Baz'}]

@app.get('/items/')
async def read_item1(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

# # Optional parameters
# @app.get("/items/{item_id}")
# async def read_item2(item_id: str, q: Union[str, None] = None):
#     if q:
#         return {"item_id": item_id, "q": q}
#     return {"item_id": item_id}

# # Query parameter type conversion
# @app.get('/items/{item_id}')
# async def read_item3(item_id: str, q: Union[str, None] = None, short: bool = False):
#     item = {'item_id': item_id}
#     if q:
#         item.update({'q': q})
#     if not short:  
#         item.update({'descripion': 'This is an amazing item that has a long description'})
#     return item

# Multiple path and query parameters
@app.get('/users/{user_id}/items/{item_id}')
async def read_user_item(
    user_id: int, item_id: str, q: Union[str, None] = None, short: bool = False):
    item = {'item_id': item_id, 'owner_id': user_id}
    if q:
        item.update({'q': q})
    if not short:
        item.update(
            {'description': 'This is an amazing item that has a long description'}
        )
    return item

# Required query parameters
# @app.get('/items/{item_id}')
# async def read_user_item2(item_id: str, needy: str):
#     item = {'item_id': item_id, 'needy': needy}
#     return item

@app.get('/items/{item_id}')
async def read_user_item3(item_id: str, needy: str, skip: int = 0, limit: Union[int, None] = None):
    item = {'item_id': item_id, 'needy': needy, 'skip': skip, 'limit': limit}
    return item