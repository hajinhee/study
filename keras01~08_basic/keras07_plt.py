from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt    
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) # shape=(20, )
y = np.array([1,2,4,3,5,7,9,3,8,12,13,17, 8,14,21, 9, 6,19,23,21]) # shape=(20, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측      
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x)

plt.scatter(x, y) # scatter 점찍다
plt.plot(x, y_predict, color='red') # plot 선을 보여준다 
plt.show() 

# epochs=100, batch_size=1 [loss]  :  22.72275733947754
# epochs=200, batch_size=1 [loss]  :  23.0115966796875
# epochs=300, batch_size=1 [loss]  :  24.1588134765625