import autokeras as ak, tensorflow as tf
from sklearn.metrics import accuracy_score

#1. load data 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#2. modeling
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)
'''
overwrite=bool : default=False -> False인 경우 동일한 이름의 기존 프로젝트가 있으면 다시 로드하고 그렇지 않으면 덮어쓴다.
max_trials=int: default=100 -> 시도할 다른 Keras 모델의 최대 수이다. max_trials에 도달하기 전에 검색이 완료될 수 있다.
'''

#3. compile, train
model.fit(x_train, y_train, epochs=5)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('[loss]: ', loss)  # [loss]: 0.03688045218586922 [accuracy_score]: 0.9889000058174133
