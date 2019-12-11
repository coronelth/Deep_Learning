import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_gaussian_quantiles

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
            print(idx)
            
        return (Yp)
        

def acc(yp,y_test):
	return (yp==y_test).mean()*100

#%% Generar los puntos
dim=3

X1, Y1 = make_gaussian_quantiles(n_features=dim, n_classes=1,n_samples=1000,cov=0.1,mean=[9.,1.,8.])
X2, Y2 = make_gaussian_quantiles(n_features=dim, n_classes=1,n_samples=1000,cov=0.1,mean=[10.,8.,2.])
X3, Y3 = make_gaussian_quantiles(n_features=dim, n_classes=1,n_samples=1000,cov=0.1,mean=[5.,2.,9.])
X4, Y4 = make_gaussian_quantiles(n_features=dim, n_classes=1,n_samples=1000,cov=0.1,mean=[3.,7.,3.])
X5, Y5 = make_gaussian_quantiles(n_features=dim, n_classes=1,n_samples=1000,cov=0.1,mean=[2.,9.,1.])

Y2=Y2+1
Y3=Y3+2
Y4=Y4+3
Y5=Y5+4

X=np.vstack((X1,X2,X3,X4,X5))
Y=np.hstack((Y1,Y2,Y3,Y4,Y5))

fig=plt.figure()
ax=fig.gca(projection='3d')
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
#plt.plot(X[:,0],x[:,1],x[:,2],'ro')

plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,s=75)
#%%

x_train=X
y_train=Y

print('--------     Datos 3D     --------')
print('x_train.shape:',x_train.shape)
print(x_train.shape[0],'train samples')

k=7
ntest=30
acc_datos=np.zeros(k)

for i in range(k):
    model = NearestNeighbor()
    model.train(x_train,y_train)


    x_testt=x_train[:ntest]
    y_testt=y_train[:ntest].ravel()

    yp_mnist=model.predict(x_testt,k).ravel()
    acc_datos[i]=acc(yp_mnist,y_testt)
    print("k:{} acc:{:.2f} n_test:{}".format((i+1),acc_datos[i],ntest))