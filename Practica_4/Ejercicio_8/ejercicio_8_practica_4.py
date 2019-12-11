# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D,MaxPooling2D,Flatten


cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train/255
x_test=x_test/255
#x_train=x_train[...,np.newaxis]
#x_test=x_test[...,np.newaxis]

model = tf.keras.models.Sequential()
model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), activation='relu',input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu'))
model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu'))
model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

y_test_t=np.zeros((y_test.shape[0],10))
y_test_t[np.arange(y_test.shape[0]),y_test]=1
y_train_t=np.zeros((y_train.shape[0],10))
y_train_t[np.arange(y_train.shape[0]),y_train]=1

history=model.fit(x_train, y_train_t, epochs=10, verbose=1, validation_data=(x_test, y_test_t))

score = model.evaluate(x_test, y_test_t, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

