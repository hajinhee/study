import autokeras as ak, tensorflow as tf

#1. 데이터 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#2. 모델
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

#3. 컴파일,훈련
model.fit(x_train, y_train, epochs=5)

#4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test,y_test)
print(results)      # [0.0270724855363369, 0.9911999702453613] 2개의 값을 던져준다. 
# max_trials=2          앞의 값은 loss,     뒤의 값은 acc이다.  max_trials의 값만큼 모델을 돌려서
# 여러개의 acc를 뽑아내고 그 중에 제일 좋은 값을 쓴다. 
model.summary()