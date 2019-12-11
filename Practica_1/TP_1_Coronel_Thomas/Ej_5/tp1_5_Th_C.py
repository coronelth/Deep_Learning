#%%
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

random_state = 159
np.random.seed(random_state)

#%%
class Clasific_lineal():
    def __init__(self,dimension,clases):
        self.w = np.random.randn(dimension,clases)*1e-4    #el dim+1 contempla el barias
        
    def score(self,x):
        s = np.dot(self.w.T,x)
    
        return s

    def predecir(self,x,y):
        
        s=self.score(x)
        pred=s.argmax(axis=0)
        return pred
    
    def loss_grad(self,x,y,reg):
        pass    

      
    def loss(self,x,y,reg):
        pass        

    def accuracy(self,x,y):
            acc=np.zeros(y.shape[0])
            acc[self.predecir(x,y)== y]=1
            acc = np.sum(acc)/(y.shape[0])*100.0
            return acc
        
      
    def fit (self,x,y,x_t,y_t,epocas,sizebatch,lr,reg=1):
                
        nro_batchs=int(y.shape[0]/sizebatch)
                
        #w_calc=np.copy(self.W)              
        loss = np.zeros(epocas) 
        val_loss = np.zeros(epocas) 
  #      dif_W = np.zeros(epocas) 
        accuracy = np.zeros(epocas)
        val_acc = np.zeros(epocas)
        for i in range(epocas):
            #print("epoca= ",i,"\t")
            
            indexrandom=x.shape[1]
            indexrandom=np.arange(indexrandom)
            np.random.shuffle(indexrandom)
            
            for j in range (nro_batchs):
                                                
                x_batch=x_train[:,indexrandom[(j*sizebatch):((j+1)*sizebatch)]]
                y_batch=y_train[indexrandom[(j*sizebatch):((j+1)*sizebatch)]]
            
                
                #s=self.score(x_batch)
                L2=0.5*(self.w*self.w).sum()
                gradiente=(self.loss_grad(x_batch,y_batch)).T + reg*self.w  
                self.w=self.w - lr *gradiente
                
                loss[i]+=self.loss(x_batch,y_batch) + L2 *reg
                accuracy[i]+=self.accuracy(x_batch, y_batch)
            loss[i]/=nro_batchs
            accuracy[i]/=nro_batchs
            #voy a tomar los 1000 primeros valores y ver cuanto le pega
            tamano_prueba=1000
            accuracy[i]=self.accuracy(x[:,:tamano_prueba],y[:tamano_prueba])
            val_loss[i]=self.loss(x_t,y_t)
            val_acc[i]=self.accuracy(x_t,y_t)
            
            print("epoca:{} acc:{:.2f} loss:{:.2f} val_acc:{:.2f} val_loss:{:.2f}".format(i,accuracy[i],loss[i],val_acc[i],val_loss[i]))
                  #,i,"\t","acc= ",accuracy[i],"\t",loss[i],"\t\n")
            #print("acc= ",accuracy[i],"\t\n")
            
        history = { "loss": loss, "acc": accuracy,"loss_val": val_loss,"val_acc" : val_acc }    
            
        return history
                
     

#%%
class Svm(Clasific_lineal):
    
    def __init__(self,dimension,clases,delta=1):
        super().__init__(dimension,clases)
        self.delta = delta
        
    def loss(self,x,y):
        sc=self.score(x)
        sy=sc[y,np.arange(y.shape[0])]
        
        asterisco = sc - sy + self.delta
        asterisco[asterisco<0] = 0
        asterisco[y,np.arange(y.shape[0])] = 0
        L=asterisco.sum(axis=0)
        loss=L.mean()
        return loss
    
    def loss_grad(self,x,y):
        
        sc=self.score(x)
        sy=sc[y,np.arange(y.shape[0])]
            
        asterisco=sc-sy+self.delta
        asterisco[asterisco<0] = 0
        asterisco[y,np.arange(y.shape[0])] = 0
        asterisco[asterisco>0] = 1
        suma=asterisco.sum(axis=0)
        asterisco[y,np.arange(y.shape[0])] = - suma
                           
        grad=np.dot(asterisco,(x.T))/(x.shape[0])
        
        return grad
    

    
# %%      
class Soft_max(Clasific_lineal):
    
    def __init__(self,dimension,clases):
        super().__init__(dimension,clases)

    def loss_grad(self,x,y,reg=1):
        sc=self.score(x)
        sc-=sc.max(axis=0)
       # sy=sc[y,np.arange(y.shape[0])]
        #sc/= sc.min(axis=1)[:, np.newaxis]
        #sc-= sc.max(axis=1)[:, np.newaxis]
        
        es=np.exp(sc)    
        #esy=es[y,np.arange(y.shape[0])]
        sum_es=np.sum(es,axis=0)
        
        #loss = log( sum_es) - sy
                
        grad=es*(1/sum_es)
        
        grad[y,np.arange(y.shape[0])] -=  1
        
        return  np.dot(grad,x.T)/x.shape[0]
    
    def loss(self,x,y):
        sc=self.score(x)
        sc-=sc.max(axis=0)
        sy=sc[y,np.arange(y.shape[0])]
        #sc/= sc.min(axis=1)[:, np.newaxis]
        #sc-= sc.max(axis=1)[:, np.newaxis]
        
        es=np.exp(sc)    
        
        sum_es=np.sum(es,axis=0)
        
        loss = np.log( sum_es) - sy
        loss=loss.mean()
        return loss

#%%
####----------- Para MNIST------------------------------#####
#
#(x_train,y_train),(x_test,y_test) = mnist.load_data()
#x_m = x_train.mean(axis=0)
#
#x_train = x_train -  x_m
#x_test = x_test - x_m
#
#dim_xt=x_train.shape
#x_train=x_train.reshape((dim_xt[0],np.prod(dim_xt[1:]))).T
#x_train=(np.vstack((np.zeros((1,dim_xt[0])),x_trainn)))
#
#
#
#dim_xte=x_test1.shape
#x_test=x_test1.reshape((dim_xte[0],np.prod(dim_xt[1:]))).T        
#x_test=np.vstack((np.zeros((1,dim_xte[0])),x_test))
#
#epocas=300   
#sizebatch=200
#lr=1e-9
#clases=10
#
#reg=1e-5
#%%
#modelo=Svm(x_train.shape[0],clases)
#history = modelo.fit(x_train,y_train,x_test,y_test,epocas,sizebatch,lr,reg)
##%%
#plt.figure()
#plt.plot(history["loss"],label= "training")
#plt.plot(history["loss_val"],label= "test")
#plt.xlabel('Epocas',fontsize=14)
#plt.ylabel('Loss',fontsize=14)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.grid(True)
#plt.legend()
#plt.title('Costo de SVM para MNIST',fontsize=14)
#plt.savefig("TP_1_Ej_Svm_loss_MNIST.png",bbox_inches='tight')
#plt.show()
#
#
#plt.figure()
#plt.plot(history["acc"],label= "training")
#plt.plot(history["val_acc"],label= "test")
#plt.xlabel('Epocas',fontsize=14)
#plt.ylabel('Accuracy',fontsize=14)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.grid(True)
#plt.legend()
#plt.title('Precisión de SVM para MNIST',fontsize=14)
#plt.savefig("TP_1_Ej_5_Svm_acc_epocas_MNIST.png",bbox_inches='tight')
#plt.show()
##%%
#modelo=Soft_max(x_train.shape[0],clases)
#history_sf = modelo.fit(x_train,y_train,x_test,y_test,epocas,sizebatch,lr,reg)
##%%
#plt.figure()
#plt.plot(history_sf["loss"],label= "training")
#plt.plot(history_sf["loss_val"],label= "test")
#plt.xlabel('Epocas',fontsize=14)
#plt.ylabel('Loss',fontsize=14)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.grid(True)
#plt.legend()
#plt.title('Costo de Soft Max para MNIST',fontsize=14)
#plt.savefig("TP_1_Ej_6_Sfm_loss_epocas_MNIST.png",bbox_inches='tight')
#plt.show()
#
#plt
.figure()
#plt.plot(history_sf["acc"],label= "training")
#plt.plot(history_sf["val_acc"],label= "test")
#plt.xlabel('Epocas',fontsize=14)
#plt.ylabel('Accuracy',fontsize=14)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.grid(True)
#plt.legend()
#plt.title('Precisión de Soft Max para MNIST',fontsize=14)
#plt.savefig("TP_1_Ej_6_Sfm_acc_epocas_MNIST.png",bbox_inches='tight')
#plt.show()

#%%

###----------- Para CIFAR 10 ------------------------------#####

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_m = x_train.mean(axis=0)

x_train = x_train -  x_m
x_test = x_test - x_m

dim_xt=x_train.shape
x_train=x_train.reshape((dim_xt[0],np.prod(dim_xt[1:]))).T
x_train=(np.vstack((np.ones((1,dim_xt[0])),x_train)))

dim_xte=x_test.shape
x_test=x_test.reshape((dim_xte[0],np.prod(dim_xte[1:]))).T        
x_test=np.vstack(((np.ones((1,dim_xte[0]))),x_test))

y_train=y_train.ravel()
y_test=y_test.ravel()

epocas=200   
sizebatch=200
#lr=1e-8
lr=1e-7
clases=10

reg=1e-5
#%%
modelo=Svm(x_train.shape[0],clases)
history = modelo.fit(x_train,y_train,x_test,y_test,epocas,sizebatch,lr,reg)
#%%
plt.figure()
plt.plot(history["loss"],label= "training")
plt.plot(history["loss_val"],label= "test")
plt.xlabel('Epocas',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.title('Costo de SVM para CIFAR 10',fontsize=14)
plt.savefig("TP_1_Ej_5_Svm_loss_epocas_CIFAR10.png",bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(history["acc"],label= "training")
plt.plot(history["val_acc"],label= "test")
plt.xlabel('Epocas',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.title('Precisión de SVM para CIFAR 10',fontsize=14)
plt.savefig("TP_1_Ej_5_Svm_acc_epocas_CIFAR10.png",bbox_inches='tight')
plt.show()
#%%
modelo=Soft_max(x_train.shape[0],clases)
history_sf = modelo.fit(x_train,y_train,x_test,y_test,epocas,sizebatch,lr,reg)
#%%
plt.figure()
plt.plot(history_sf["loss"],label= "training")
plt.plot(history_sf["loss_val"],label= "test")
plt.xlabel('Epocas',fontsize=14)
plt.ylabel('Loss',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.title('Costo de Soft Max para CIFAR 10',fontsize=14)
plt.savefig("TP_1_Ej_6_Sfm_loss_epocas_CIFAR10.png",bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(history_sf["acc"],label= "training")
plt.plot(history_sf["val_acc"],label= "test")
plt.xlabel('Epocas',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend()
plt.title('Precisión de Soft Max para CIFAR 10',fontsize=14)
plt.savefig("TP_1_Ej_6_Sfm_acc_epocas_CIFAR10.png",bbox_inches='tight')
    
    
