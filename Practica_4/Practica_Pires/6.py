# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, _), (x_test, _) = mnist.load_data()
y_train=x_train/255
y_test=x_test/255
x_train = np.clip(0.5*np.random.randn(x_train.shape[0],x_train.shape[1],x_train.shape[2])+(x_train),0,1)
x_test = np.clip(0.5*np.random.randn(x_test.shape[0],x_test.shape[1],x_test.shape[2])+(x_test),0,1)
y_train=y_train.reshape((y_train.shape[0],784))
y_test=y_test.reshape((y_test.shape[0],784))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(300, activation='relu'),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(300, activation='relu'),
  tf.keras.layers.Dense(784, activation='sigmoid'),
])


model.compile(optimizer='adam',loss='MSE')
model.fit(x_train, y_train, epochs=15)
decode=model.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.title('y_true')
    plt.imshow(y_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(3, n, i + 1+n)
    plt.title('with noise')
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 +2* n)
    plt.title('decode')
    plt.imshow(decode[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('ej6mnist.png')
plt.show()