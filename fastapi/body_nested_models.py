from email.mime import image
from typing import Union, List, Set, Dict
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()

class Image(BaseModel):
    # url: str
    url: HttpUrl  # url 필드의 str을 파이던틱 HttpUrl 로 대체 가능
    name: str

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    # tags: List[str] = []  # python 3.6 이상
    # tags: list[str] = []  # python 3.9 이상
    tags: Set[str] = set()  # python 3.6 이상
    # tags: set[str] = set()  # python 3.9 이상
    # image: Union[Image, None] = None
    image: Union[List[Image], None] = None  # Pydantic 모델을 목록, 세트 등의 하위 유형으로 사용할 수도 있다.

# Deeply nested models
class Offer(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    items: List[Item]

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results

@app.post("/offers/")
async def create_offer(offer: Offer):
    return offer

# Bodies of pure lists
@app.post("/images/multiple/")
async def create_multiple_images(images: Image):
    return images

# Bodies of arbitrary dicts
@app.post("/index-weights/")
async def create_index_weights(weights: Dict[int, float]):
    return weights