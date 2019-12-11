#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Sequential

#%% Creación del modelo 
def model_create(opt):
  model = Sequential()
  model.add(Input(shape=(x_train.shape[1],)))
  model.add(Dense(32, activation = 'elu'))
  model.add(Dense(32, activation = 'relu'))
  model.add(Dense(16, activation = 'elu'))
  model.add(Dense(1, activation = 'sigmoid'))
  model.compile(loss ='binary_crossentropy',
                optimizer = opt, metrics = ['acc'])
 
  return model

#%%   Entreamiento del modelo con StratifiedKFold para un modelo dado
def train_with_kfold(x_train,y_train,opt,nsplit,shuffle,random_state):
    acu  = []
    loss = []
    for train_index,test_index in StratifiedKFold(
            n_split,shuffle,random_state).split(x_train, y_train):

        x_t  = x_train[train_index]
        x_tt = x_train[test_index]
        y_t  = y_train[train_index]
        y_tt = y_train[test_index]
      
        model= model_create(opt)
        history=model.fit(x_t, y_t,epochs=nepoch,validation_data=(x_tt, y_tt))
        loss.append(history.history['val_loss'])
        acu.append(history.history['val_acc'])
      
    return np.array(acu),np.array(loss)

#%% Funcion que obtiene los parámetros para realizar la gráfica pedida
def obtenc_param(param):
    return np.max(param,axis=0),np.min(param,axis=0),np.mean(param, axis=0)
   
#%% Obtención de los datos
dataset=np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
x_train=dataset[:,0:8]
y_train=dataset[:,8]

#%%   Datos para entrear los dos modelos, difieren en el optimizados
n_split = 5
nepoch  = 500
shuffle = True
random_state = 15

opt_1  = 'adam'
opt_2   = 'SGD'

acu_1,loss_1 = train_with_kfold(x_train,y_train,
                               opt_1,n_split,shuffle,random_state)

acu_2,loss_2 = train_with_kfold(x_train,y_train,
                               opt_2,n_split,shuffle,random_state)

#%% Obtensión de los valores para graficar los modelos
#adam
max_acu_1  ,min_acu_1  ,mean_acu_1  = obtenc_param(acu_1)
max_loss_1 ,min_loss_1 ,mean_loss_1 = obtenc_param(loss_1)
#SGD
max_acu_2  ,min_acu_2  ,mean_acu_2  = obtenc_param(acu_2)
max_loss_2 ,min_loss_2 ,mean_loss_2 = obtenc_param(loss_2)

#%% Graficos para el modelo con opt = adam
plt.figure()
#plt.plot(min_acu_1,'k--',label='min_acu')
#plt.plot(max_acu_1,'k',label='max_acu')
plt.fill_between(np.arange(nepoch),min_acu_1,max_acu_1)
plt.plot(mean_acu_1,'r',label='mean_acu')
plt.title('Accuracy on Training performance with optimizer adam',fontsize=14)
plt.xlabel('Number of epochs',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("TP_4_ej_5_acc_opt_adam.png",bbox_inches='tight')
plt.show()

plt.figure()
#plt.plot(min_loss_1,'k--',label='min_loss')
#plt.plot(max_loss_1,'k',label='max_loss')
plt.fill_between(np.arange(nepoch),min_loss_1,max_loss_1)
plt.plot(mean_loss_1,'r',label='mean_loss')
plt.title('Loss on Training performance with optimizer adam',fontsize=14)
plt.xlabel('Number of epochs',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("TP_4_ej_5_loss_opt_adam.png",bbox_inches='tight')
plt.show()

#%% Graficos para el modelo con opt = SGD
plt.figure()
#plt.plot(min_acu_2,'k--',label='min_acu')
#plt.plot(max_acu_2,'k',label='max_acu')
plt.fill_between(np.arange(nepoch),min_acu_2,max_acu_2)
plt.plot(mean_acu_2,'r',label='mean_acu')
plt.title('Accuracy on Training performance with optimizer SGD',fontsize=14)
plt.xlabel('Number of epochs',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("TP_4_ej_5_acc_opt_SGD.png",bbox_inches='tight')
plt.show()

plt.figure()
#plt.plot(min_loss_2,'k--',label='min_loss')
#plt.plot(max_loss_2,'k',label='max_loss')
plt.fill_between(np.arange(nepoch),min_loss_2,max_loss_2)
plt.plot(mean_loss_2,'r',label='mean_loss')
plt.title('Loss on Training performance with optimizer SGD',fontsize=14)
plt.xlabel('Number of epochs',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("TP_4_ej_5_loss_opt_SGD.png",bbox_inches='tight')
plt.show()
