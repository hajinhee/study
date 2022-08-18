from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델링 
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.25) 

model.load_weights('./_save/keras25_3_save_weights.h5')
# save_weights, load_weights는 일반 save와 다르게 model = Sequential()과 model.compile()해줘야 사용이 가능하다
# fit단계 전에 하냐 후에 하냐에 따라 차이가 있지만 후에 쓰는게 바른 방법이고 그래야 값이 저장된다.

#4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
loss :  72.95464324951172
r2스코어 :  0.12715872536555994
'''