import autokeras as ak, tensorflow as tf

#1. load data 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

#2. modeling
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=5
)

#3. compile, train
model.fit(x_train, y_train, epochs=3)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('[loss]: ', loss[0], '[accuracy_score]: ', loss[1])    
