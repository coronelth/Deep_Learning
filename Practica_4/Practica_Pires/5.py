# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
n_split=5
nepoch=20

dataset=np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
x_train1=dataset[:,0:8]
y_train1=dataset[:,8]

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(24, input_shape=(x_train1.shape[1],) , activation = 'relu'))
  model.add(tf.keras.layers.Dense(24, activation = 'relu'))
  model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
  model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['acc'] )
 
  return model

n_split=5
i=0 
loss=np.empty([5,nepoch])
acu=np.empty([5, nepoch])
for train_index,test_index in KFold(n_split).split(x_train1):
  x_train,x_test=x_train1[train_index],x_train1[test_index]
  y_train,y_test=y_train1[train_index],y_train1[test_index]
  
  model=create_model()
  history=model.fit(x_train, y_train,epochs=nepoch, validation_data=(x_test, y_test))
  loss[i,:]=history.history['val_loss']
  acu[i,:]=history.history['val_acc']
  i=i+1


minloss=np.min(loss,axis=0) 
maxloss=np.max(loss,axis=0)
maxacu=np.max(acu,axis=0)
minacu=np.min(acu,axis=0)

plt.figure(1)
plt.plot(minloss,label='minloss')
plt.plot(maxloss,label='maxloss')
plt.fill_between(np.arange(nepoch),minloss,maxloss)
plt.plot((minloss+maxloss)/2,label='mean')
plt.legend()
plt.close

plt.figure(2)
plt.plot(minacu,label='minacu')
plt.plot(maxacu,label='maxacu')
plt.fill_between(np.arange(nepoch),minacu,maxacu)
plt.plot((minacu+maxacu)/2,label='mean acu')
plt.legend()
plt.close
#model.evaluate(x_test,  y_test, verbose=2)
#predictions = model.predict(x_test)
#predictions=predictions.ravel()
#loss=(predictions-y_test)**2
#loss=loss.mean()