import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

#x = np.array(1,2,3,4,5,6,7)
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])     #위의 값을 timesteps -> 3으로 해서 쪼개주면 아래와 같이 된다.

y = np.array([4,5,6,7])                             

print(x.shape, y.shape)         # (4, 3) (4,) 

#x.reshape = (batch, timesteps, feature)
#x.reshape = (행, 열, 몇개씩 자르는지!!!)       어차피 근데 여기서의 배치값은 데이터 원래의 길이에서 timesteps있으니까 알아서 구해진다.
# 내용물과 순서 바꾸면 안됌. 2차원 -> 3차원 변환해야 RNN 적용해서 쓸수있다.

x = x.reshape(4,3,1)    # 4개의 행이되고(7을 3으로 겹쳐서 나눴으므로),3(timesteps값),1씩 슬라이스하겠다. -> 이걸 Rnn에 넣어줘야 RNN연산되지
# 이걸 cnn에 넣으면 cnn방식으로 되어버린다. 근데 차원 달라서 에러뜰듯. 

#print(x)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, activation='linear',input_shape=(3,1))) #행은 넣어주지않는다.(row값)
model.add(Dense(10))        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         

# activation의 역할은 다음레이어로 전달하는 값을 한정시켜줌. 아무곳에나 다 넣을수있다.
# RNN도 결국 하나의 방법일뿐 근본은 결국 같다.
# 근본은 결국 데이터의 행 * 수치값인데 그걸 접었다 폈다하면서 2차원 3차원 4차원이라 하면서 조정해 주는것.
# 행값만 그대로 주면서 그 뒤의 데이터를 손질해서 RNN방식으로 DNN방식으로 또 CNN방식으로 모델링해가면서 가장 좋은 수치가 나오는걸 쓰면 된다.
# np와 pd의 차이 *****   -> 인공지능의 모든 수치는 numpy, pandas도 numpy로 구성되어 있는데 추가 기능이 있다?

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  #여기서의 batch_size는 행을 몇개씩 때려박을건지를 의미 <-- 
# 원데이터 x는 나의 편의에 따라 4,3행의 행렬로 값이 중복된 채로 바뀌었고 1epoch에 이 데이터를 한바퀴 다 돌리는데
# batch_size로 행을 몇개씩 집어넣어서 돌릴건지 결정해준다.  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7]).reshape(1,3,1)
result = model.predict(y_pred)   # input_shape를 원래입력값과 똑같이맞춰줘야한다. 
print(result)

