from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import EarlyStopping    
from icecream import ic

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# monitor: 어떤 값을 기준으로 하여 훈련 종료를 결정한 것인가
# patience: 기쥰이 되는 값이 연속으로 몇 번 이상 향상되지 않을 때 종료할 것인가
'''
patience가 0이 아닌 경우 주의해야할 사항이 있다. 
위 예제를 상기해보자. 만약 20번째 epoch까지는 validaion loss가 감소하다가 21번째부턴 계속해서 증가한다고 가정해보자. 
patience를 5로 설정하였기 때문에 모델의 훈련은 25번째 epoch에서 종료할 것이다. 
그렇다면 훈련이 종료되었을 때 이 모델의 성능은 20번째와 25번째에서 관측된 성능 중에서 어느 쪽과 일치할까? 
안타깝게도 20번째가 아닌 25번째의 성능을 지니고 있다. 위 예제에서 적용된 early stopping은 훈련을 언제 종료시킬지를 결정할 뿐이고, 
Best 성능을 갖는 모델을 저장하지는 않는다. 따라서 early stopping과 함께 모델을 저장하는 callback 함수를 반드시 활용해야만 한다.
'''

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) # 모델을 저장하는 callback 함수
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초') 

#4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # 38.619388580322266

y_predict = model.predict(x_test)
print("최적의 로스값 : ", y_predict)

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2) # 0.4733751866818521

plt.figure(figsize=(9,6)) 
plt.plot(model.history['loss'], marker=".", c='red', label='loss')
plt.plot(model.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()