#%%
#import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#%%

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_test=x_test/x_train.max()
x_train = x_train/x_train.max()

y_test=y_test/y_train.max()
y_train=y_train/y_train.max()



#%%

model = Sequential()
model.add(Input(shape=(13,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam',loss='MSE')
model.fit(x_train, y_train, epochs=1000,batch_size=100,shuffle=True)

loss= model.evaluate(x_test,  y_test, verbose=1)
predictions = model.predict(x_test).ravel()
#%%
plt.scatter(y_test, predictions)
p=sp.polyfit(y_test, predictions, 1)
plt.plot(np.linspace(0,1,10), np.linspace(0,1,10)*p[0]+p[1])
plt.xlim(0, 1)
plt.ylim(0, 1)
#print((np.argmax(predictions)==y_test).mean())
