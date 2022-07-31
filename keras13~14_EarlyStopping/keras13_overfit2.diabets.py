
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

#1 데이터
datasets = load_diabetes() 
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)
'''
랜덤스테이트를 지정하지 않으면 train할 때마다 다른 랜덤값으로 나오기 때문에 조정한 파라미터를 비교할 수 없다. 
'''

#2. 모델링 
model = Sequential()
model.add(Dense(20, input_dim=10))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

start = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2) 
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

#5. 예측
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

# epochs=100, batch=1  [loss] :  2954.56103515625  [r2스코어] :  0.4423411741455825
# (layer수, node수 줄이고) epochs=100, batch=1  [loss] :  2982.88623046875  [r2스코어] :  0.43699489865850183
# (layer수, node수 더 줄이고) epochs=50, batch=1  [loss] :  2942.2119140625  [r2스코어] :  0.5190015735821165

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker=".", c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() 
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

'''
그림도표를 보면서 과적합이 어떨때 일어나는지생각해보자..?
훈련을 많이 한다고 좋은게 아니다. 그래프를 보면 값들이 줄어들었다가 팡 튀고 줄어들었다가 팡 튀고 한다.
계속 여러번 돌려보면서 loss와 var_loss 격차가 많이 줄어가는걸 보면서 epoch량을 조절한다.
val_loss가 최저점이다라는 말의 뜻은 y = wx + b 예측을 가장 잘했다. 
편하게 최저점을 구하는 방법이 뭐가 있을까...     무한루프 돌리고 조건식을 설정해준다.
일정 최저값을 찍고 특정횟수의 유예를 준다. 그 유예동안 최저값 갱신이 안되면 다시 끊고 갱신이 되면 다시 루프돌린다.
'''
