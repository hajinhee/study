from enum import Enum
from fastapi import FastAPI

app = FastAPI()

# 장고의 모델과 비슷한 역할
class ModelName(str, Enum):  # 비슷한 종류의 상수들을 묶어놓기 위해 Enum(열거형) 사용
    lenet = 'lenet'
    alexnet = 'alexnet'
    resnet = 'resnet'

@app.get('/medels/{model_name}')
async def get_model(model_name: ModelName):  
    if model_name is ModelName.lenet:
        return {'model_name': model_name, 'message': 'LeCNN all the images'}

    if model_name.value == 'alexnet':
        return {'model_name': model_name, 'message': 'Deep Learning FTW!'}

    return {'model_name': model_name, 'message': 'Have some residuals'}

@app.get('/files/{files_path:path}')
async def read_file(file_path: str):
    return {'file_path': file_path}
