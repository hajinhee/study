from tensorflow.keras.models import Sequential
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

#2. 모델링 
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_11_MCP.hdf5')
'''
es에서 restore_bets_weights = True, False 하든말든 mcp는 나름대로 save_best_only를 계속 수행하고 값을 저장한다.
save_weights_only = True 했을경우 load_weights해서 사용
'''

hist = model.fit(x_train, y_train, epochs=500, batch_size=8, validation_split=0.25, callbacks=[es, mcp]) 

print("-------------------------------------------")
print(hist)  # 자료형이 나온다.
print("-------------------------------------------")
print(hist.history)  # loss 값과 var_loss값이 dic형태로 저장되어 있다. epoch 값만큼의 개수가 저장되어 있다 ->> 1epoch당 값을 하나씩 다 저장한다.
print("-------------------------------------------")
print(hist.history['loss'])  # hist.history에서 loss키 값의 value들을 출력해준다.
print("-------------------------------------------")
print(hist.history['val_loss'])  # hist.history var_loss키 값의 value들을 출력해준다.
print("-------------------------------------------")


#4. loss 시각화
plt.figure(figsize=(9,6)) 
plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()


#5. 모델저장
model.save("./_save/keras26_11_save_MCP.h5")  # 여기서 저장하는 모델은 es에서 주는 가중치가 저장된 모델

#6. 평가 , 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

'''
[loss] :  24.32086753845215
[r2스코어] :  0.7090210235289973
'''