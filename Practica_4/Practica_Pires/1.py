# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_test=np.hstack((x_test,np.ones((x_test.shape[0],1))))/x_train.max()
x_train=np.hstack((x_train,np.ones((x_train.shape[0],1))))/x_train.max()

y_test=y_test/y_train.max()
y_train=y_train/y_train.max()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(14,)),
  tf.keras.layers.Dense(1),
])
model.compile(optimizer='adam',loss='MSE',metrics=['mae'])
model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test,  y_test, verbose=2)
predictions = model.predict(x_test)
predictions=predictions.ravel()
loss=(predictions-y_test)**2
loss=loss.mean()
#print((np.argmax(predictions,axis=1)==y_test).mean())