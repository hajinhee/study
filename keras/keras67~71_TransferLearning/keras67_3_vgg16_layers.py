from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16
import pandas as pd

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = Sequential()
model.add(vgg16)
# 분류기 구현
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

pd.set_option('max_colwidth', -1)  # 출력할 각 컬럼의 길이 최대로
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
'''
                                                              Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x0000024F9D91AEE0>      vgg16      True
1  <keras.layers.reshaping.flatten.Flatten object at 0x0000024F9D920B50>  flatten    True
2  <keras.layers.core.dense.Dense object at 0x0000024F9D920FD0>           dense      True
3  <keras.layers.core.dense.Dense object at 0x0000024FA494BC40>           dense_1    True
'''

model.trainable = False  # 앞으로 있을 추가 학습에서 가중치가 변경되지 않도록 고정 
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

'''
                                                              Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x000001BCB3C99EB0>      vgg16      False
1  <keras.layers.reshaping.flatten.Flatten object at 0x000001BCB3CA0B20>  flatten    False
2  <keras.layers.core.dense.Dense object at 0x000001BCB3CA0FA0>           dense      False
3  <keras.layers.core.dense.Dense object at 0x000001BCB403DC10>           dense_1    False
'''