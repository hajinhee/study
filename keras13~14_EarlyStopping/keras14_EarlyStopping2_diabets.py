from tensorflow.keras.models import Sequential          # 신경망 모델링할 모델의 종류 Sequential
from tensorflow.keras.layers import Dense               # 모델링에서 사용할 레이어의 종류 Dense 
from sklearn.datasets import load_diabetes              # 싸이킷런 라이브러리의 datasets클래스의 diabets함수 불러옴
from sklearn.model_selection import train_test_split    # 데이터를 train과 test로 0.0~1.0 사이의 비율로 분할 및 랜덤분류 기능
from sklearn.metrics import r2_score                    # y_predict값과 y_test값을 비교하여 점수매김. 0.0~1.0 및 - 값도 나옴
import matplotlib.pyplot as plt                         # 데이터 시각화
import time
from tensorflow.keras.callbacks import EarlyStopping    # training 조기종료를 도와주는 기능 여러 옵션들이 있다.
from icecream import ic

#1. 데이터 로드
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# ic(x.shape) # (442, 10)
# ic(datasets.feature_names)  # 컬럼명 ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR) 

#2. 데이터 정제
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42) 

#3. 모델링
# 이 단계를 여러번 반복해서 좋은 모델을 찾아야 한다.  
model = Sequential()
model.add(Dense(50, input_dim=10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#4. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, baseline = None, restore_best_weights=True)
'''val_loss를 관측하고 50번안에 최저값이 갱신되지 않으면 훈련을 중단하고 가장 좋았을때의 "weights"값을 복원하여 기록(?)합니다.
컴파일해보면 마지막에 Restoring model weights from the end of the best epoch. 라는 메시지를 출력시켜준다. 안심할수 있다.
baseline = None, 모델이 달성해야하는 최소한의 기준값, 정확도를 선정합니다. 정확도를 측정하는 기준은 0.0~1.0 
각각 True와 False를 넣었을 때 큰 차이가 없는 이유: EarlyStopping은 최적의 weights값을 복원해서 저장한다. <-- 기록하고 저장해서 evaluate 할때 최적의 값으로 계산한다.
값을 저장하려면 ModelCheckpoint 함수를 써야한다.'''

hist = model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.2, callbacks=[es]) # loss와 똑같이 관측하기 위해 일단 저장.

#5. 평가
loss = model.evaluate(x_test, y_test)  # 컴파일 단계에서 도출된 weight 및 bias값에 xy_test를 넣어서 evaluate된 값을 loss에 저장
print('평가만 해본 후 나온 loss의 값 : ', loss) # val_loss와 loss의 차이가 적을수록 validation data가 더 최적의 weights를 도출시켜줘서 실제로 평가해봐도 차이가 적게 나온다는 말이므로 차이가 적을수록 좋다.

#6. 예측
y_predict = model.predict(x_test) # x_test 값에 대한 라벨 예측

r2 = r2_score(y_test, y_predict) # 예측값과 실제값 비교
print('r2스코어는', r2)

# epochs=100, batch=1  [loss] :  2891.475341796875  [r2스코어] :  0.45424827831185366
# (layer수, node수 줄인 후)epochs=300, batch=3  [loss] :  3038.18212890625  [r2스코어] :  0.4265580869898553
# (layer수, node수 줄인 후)epochs=200, batch=1  [loss] :  2989.892822265625  [r2스코어] :  0.4356724637659378

plt.figure(figsize=(6, 4)) 
plt.plot(hist.history['loss'], marker=".", c='red', label='loss') 
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')  
plt.grid() 
plt.title('loss')   
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend(loc='upper right') 
plt.show()


