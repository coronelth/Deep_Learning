from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10

from scipy import stats

import numpy as np

random_state = 159
np.random.seed(random_state)
#%%

class NearestNeighbor:
    def _init_(self):
        self.X = None
        self.Y = None	#grupo al que pertenece

    def train(self, X, Y):
        self.im_shape = X.shape[1:]
        self.X = np.reshape(X,(X.shape[0],np.prod(self.im_shape)))
        self.Y = Y	
    def predict(self,X,k=3):
        assert self.X is not None, 'Trai method needs to be call first'
        Yp = np.zeros(X.shape[0],np.uint8)
        
        for idx in range (X.shape[0]):
            #norm1 = np.linalg.norm(self.X - X[idx].reshape((1,self.X.shape[1]),axis=-1)
            norm= np.sqrt(np.sum((self.X - X[idx].reshape((1,self.X.shape[1])))**2,axis=1))
            #print(norm-norm1) 
            id_k=norm.argsort()[:k]
            k_vect=self.Y[id_k]         
            
            m = stats.mode(k_vect)
            Yp[idx]=m[0][0]
            #print(idx)
            
        return (Yp)
        

def acc(yp,y_test):
	return (yp==y_test).mean()*100.0

#%%
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print('--------     MNIST     --------')
print('x_train.shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')


k=7
ntest=50
acc_mnist=np.zeros(k)

for i in range(k):

    model = NearestNeighbor()
    model.train(x_train,y_train)

    x_testt=x_test[:ntest]
    y_testt=y_test[:ntest].ravel()

    yp_mnist=model.predict(x_testt,k).ravel()
    acc_mnist[i]=acc(yp_mnist,y_testt)

    print("k:{} acc:{:.2f} n_test:{}".format((i+1),acc_mnist[i],ntest))

#%%

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

print('--------     CIFAR 10     --------')
print('x_train.shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
k=5
ntest=30
acc_cif=np.zeros(k)

for i in range(k):

    model = NearestNeighbor()
    model.train(x_train,y_train)


    x_testt=x_test[:ntest]
    y_testt=y_test[:ntest].ravel()

    yp_cif=model.predict(x_testt,k).ravel()
    acc_cif[i]=acc(yp_cif,y_testt)
    print("k:{} acc:{:.2f} n_test:{}".format((i+1),acc_cif[i],ntest))
#    print('accuracy:',acc_cif ,'% ')
#    print('termine')
#    
#%%

















##----------    Cosas que no salieron -------------------------

#acuracy = acc(k,x_train,y_train)
#print(acuracy)
#yp=model.predict(x_test[:10])
#print(yp)



#    def knn(self,X,Y,k):
#        assert self.X is not None, 'Trai method needs to be call first'
#        k_nearest =np.zeros(k) #entregara un array con los vecinos mas cercanos
#        clases_X = np.zeros(X.shape[0],np.uint8) #entregaremos la clase ganadora
#        
#        for idx in range (X.shape[0]):
#            #(ind)=np.argsort(X)
#            for j in range (k_nearest.shape[0]):
#                (k_nearest[j],idmin,idmax)= self.predict(X[idx])
#                X[idmin]=X[idmax]
#                Y[idmin]=Y[idmax]
#            m = stats.mode(k_nearest)
#            clases_X[idx] = m
#            print(m)
#            return clases_X    
#            
#(x_train,y_train),(x_test,y_test) = cifar10.load_data()
#
#
#k=1
#def NKK(k,X,Y):
#    k_nearest = np.zeros(k)
#    clases_X=np.zeros(shape.X[0])
#    for idx in range (X.shape[0]):
#        for j in range (k_nearest.shape[0]):
#            (k_nearest[j],idmin,idmax)= self.predict(X)
#            X[idmin]=X[idmax]
#            Y[idmin]=Y[idmax]
#        m = stats.mode(k_nearest)
#        clases_X[idx] = m
#        print(m)
#        return clases_X    
#def acc(k,X,Y):
#	a=np.zeros(k,np.uint8)
#	a = int(np.all(Y == KNN(k,X,Y)))
#	print(acc)
		

#	for i in range (X.shape[0]):
#		if Y[i]==KNN(k,X[i],Y[i]):
#			a +=1
