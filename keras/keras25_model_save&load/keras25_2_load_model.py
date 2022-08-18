from tensorflow.keras.models import Sequential, Model, load_model 
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

'''
과적합 예제
pochs를 많이 준다고해서 무조건 좋은게 아니다. 오히려 과적합이 걸려서 loss, val_loss값이 튈 수 있다.
'''

#1 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 불러오기
model = load_model('./_save/keras25_1_save_model.h5')
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.25) 

#4. 평가 , 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

'''
[loss] :  25.013750076293945
[r2스코어] :  0.7007313984637353
'''

