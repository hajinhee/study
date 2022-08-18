from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array(range(100)) # shape=(100, )  
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
'''         
y = np.array(range(1,101)) # shape=(100, )      
'''
[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
  91  92  93  94  95  96  97  98  99 100] 
'''    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=66)
'''
random_state -> 훈련을 반복할 때마다 매번 다른 값이 나오면 훈련이 제대로 되지 않는다. 동일한 값이 나오도록 정수를 넣어준다.
'''

#2. 모델링
model =  Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일 , 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss: ', loss) 
result = model.predict([150])
print('[150]의 예측값 : ', result)

# epochs=200, batch=1 [150]의 예측값 :  [[150.94412]]
# epochs=100, batch=1 [150]의 예측값 :  [[151.00246]]
# epochs=300, batch=1 [150]의 예측값 :  [[150.99939]]
