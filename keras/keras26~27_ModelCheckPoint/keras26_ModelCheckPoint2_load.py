from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# #2. 모델링 
# model = Sequential()
# model.add(Dense(40, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam') 
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_2_MCP.hdf5')
# hist = model.fit(x_train, y_train, epochs=50, batch_size=8, validation_split=0.25, callbacks=[es,mcp]) 

# print("-------------------------------------------")
# print(hist.history['val_loss']) # hist.history var_loss키 값의 value들을 출력해준다.
# print("-------------------------------------------")


# # restore_best_weights=True 한 경우
# model.save("./_save/keras26_2_save_MCP.h5")  
# model = load_model('./_ModelCheckPoint/keras26_2_MCP.hdf5')
# model = load_model('./_save/keras26_2_save_MCP.h5')  # 확인해보니 값이 동일하게 나온다.

# restore_best_weights=False 한 경우
model = load_model('./_save/keras26_11_save_MCP.h5')            

#4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
loss :  24.32086753845215
r2스코어 :  0.7090210235289973
'''