import autokeras as ak, tensorflow as tf

#1. load data 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#2. modeling
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2  # max_trials만큼 모델을 돌려서 여러 개의 'accuracy'를 뽑아내고 그 중에 제일 좋은 값을 사용
)

#3. compile, train
model.fit(x_train, y_train, epochs=5)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('[loss]: ', loss[0], '[accuracy_score]: ', loss[1])    


