from sklearn.datasets import fetch_covtype              # 데이터 입력받음
from tensorflow.keras.models import Sequential          # 모델링 모델 -> Sequential 
from tensorflow.keras.layers import Dense               # 레이어 -> Dense
import numpy as np                                      # 데이터 numpy이용해서 처리
from sklearn.model_selection import train_test_split    # 데이터 정제작업 도와주는 함수
from tensorflow.keras.callbacks import EarlyStopping    # 데이터 훈련 자동으로 멈추게해주는 함수
# 원핫인코딩 방법
from tensorflow.keras.utils import to_categorical       # y라벨 값을 0부터 순차적으로 끝까지 변환해준다  
from sklearn.preprocessing import OneHotEncoder         # y라벨 값을 유니크값만큼만 변환해준다        
from pandas import get_dummies                          # y라벨 값을 유니크값만큼만 변환해준다(라벨값이랑 인덱스정보가 들어가 있다)


#1. 데이터 
datasets = fetch_covtype()
print(datasets.DESCR)          
print(datasets.feature_names) 
'''
['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 
'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 
'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 
'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 
'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 
'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']
'''                                                      

x = datasets.data   
y = datasets.target

print(type(y)) # <class 'numpy.ndarray'>
print(x.shape, y.shape)  # (581012,54) (581012,) 
print(np.unique(y))  # [1 2 3 4 5 6 7] -> 클래스가 7개로 다중분류 모델

# catogorical은 (581012, 8) -> 0 1 2 3 4 5 6 7 8
# 싸이킷런의 OneHotEncoder은 (581012, 7)
# 판다스의 get_dummies는 변환과 더불어 y값을 출력해보면 행의 개수와 y칼럼은 유니크별로 깔끔하게 다 정리까지 해준다
# (581012,) -> (581012, 7) or (581012, 8)로 softmax 넣으면 되겠지만 상식적으로 8 선택해야 할 이유가 없다 

en = OneHotEncoder(sparse=False)         
y = en.fit_transform(y.reshape(-1, 1))   

print(y.shape)  # (581012, 7) 로 바뀐 것 확인      

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) 


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(200, activation='linear', input_dim=54))  # 회귀모델 activation = linear (default값)    
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(30))   
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))  # 이진분류 sigmoid 다중분류 softmax 
 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 이진분류 binary_categorical_crossentropy 다중분류 categorical_crossentropy   

es = EarlyStopping(monitor = "val_loss", patience=50, mode='min', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, validation_split=0.2, callbacks=[es])  # batch_size(default) = 32


#4. 평가
loss = model.evaluate(x_test, y_test)   
print('loss : ', loss[0])   
print('accuracy : ', loss[1])  
 
#5. 예측
results = model.predict(x_test[:15])
print(y_test[:15])
print(results)