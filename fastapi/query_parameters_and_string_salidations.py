from typing import Union
from fastapi import FastAPI, Query
from pydantic import Required

app = FastAPI()

# Additional validation
# @app.get('/items/')
# async def read_items(q: Union[str, None] = Query(default=None,  min_length=3, max_length=10)):
#     results = {'items': [{'item_id': 'Foo'}, {'item_id': 'Bar'}]}
#     if q:
#         results.update({'q': q})
#     return results

# Add regular expressions
# @app.get('/items/')
# async def read_items(
#     q: Union[str, None] = Query(
#         default=None, min_length=3, max_length=50, regex='^fixedquery$'
#     )
# ):
#     results = {'items': [{'item_id': 'Foo'}, {'item_id': 'Bar'}]}
#     if q:
#         results.update({'q': q})  # type: ignore
#     return results

# Required with Ellipsis (...)
@app.get('/items/')
# async def read_items(q: str = Query(default=..., min_length=3)):
async def read_items(q: Union[str, None] = Query(default=..., min_length=3)):
    results = {'items': [{'item_id': 'Foo'}, {'item_id': 'Bar'}]}
    if q:
        results.update({'q': q})  # type: ignore
    return results


# Use Pydantic's Required instead of Ellipsis (...)
@app.get("/items/")
async def read_items2(q: str = Query(default=Required, min_length=3)):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})  # type: ignore
    return results

# Query parameter list / multiple values
