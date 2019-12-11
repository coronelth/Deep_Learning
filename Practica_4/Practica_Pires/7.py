# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D,MaxPooling2D,Flatten, Dropout

mnist = tf.keras.datasets.mnist
clases=10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train/255.0
x_test = x_test/255.0

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
y_test_t=np.zeros((y_test.shape[0],clases))
y_test_t[np.arange(y_test.shape[0]),y_test]=1
y_train_t=np.zeros((y_train.shape[0],clases))
y_train_t[np.arange(y_train.shape[0]),y_train]=1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(100, activation='relu'),
  
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(x_train, y_train_t, epochs=10, verbose=1, validation_data=(x_test, y_test_t))

plt.figure(1)
plt.plot(history.history['val_loss'],label='loss test')
plt.plot(history.history['loss'],label='loss train')
plt.legend()
plt.savefig('7denseloss.png')
plt.close

plt.figure(2)
plt.plot(history.history['val_acc'],label='acu test')
plt.plot(history.history['acc'],label='acu train')
plt.legend()
plt.savefig('7denseacu.png')
plt.close


x_train2 = x_train.reshape(-1, 28, 28, 1)
model = tf.keras.models.Sequential()
model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),
                 padding='same', activation='relu',input_shape=x_train2.shape[1:]))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1),
                 padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit(x_train2, y_train_t, epochs=10, verbose=1, validation_data=(x_test, y_test_t))
