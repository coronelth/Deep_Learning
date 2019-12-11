import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_gaussian_quantiles
#%%

#---- Declaración de la dimensión de los puntos, grupos de puntos y clases de puntos
#dim=3
#p_distrib_datos=4
#k_clases=5
#N=100                     #número de puntos por grupo de puntos

#dim=3
#n_classes=1
#n_samples=np.array([350,1300,500,600,800])
#means=np.array([[1.,9.,9.],[3.,7.,1.],[5.,2.,9.],[7.,8.,2.],[9.,4.,8.]]) 
#cov=np.ones(dim)/10.0
#
#datos=[make_gaussian_quantiles(mean=means[i],cov=cov,n_samples=n_samples[i],n_features=dim,n_classes=n_classes) for i in range (len(n_samples))]
##%% obtener los puntos
#
#x= [f[0] for f in datos]
#l=sum(n_samples)
#puntos_datos=np.split(np.concatenate(x).ravel(),l)
#
##%% Graficar los puntos para ver que esten bien
#n=len(n_samples)
#colors=plt.cm.jet(np.linspace(0,1,n))
#fig=plt.figure()
#ax=fig.gca(projection='3d')
#plt.xlabel('X', size=14)
#plt.ylabel('Y', size=14)
#plots_puntos=[plt.plot(x[r][:,0],x[r][:,1],x[r][:,2],'ro',c=colors[r]) for r in range(len(n_samples))]

#%% Generar los puntos
dim=3

X1, Y1 = make_gaussian_quantiles(n_features=3, n_classes=1,n_samples=1000,cov=0.1,mean=[9.,1.,8.])
X2, Y2 = make_gaussian_quantiles(n_features=3, n_classes=1,n_samples=1000,cov=0.1,mean=[10.,8.,2.])
X3, Y3 = make_gaussian_quantiles(n_features=3, n_classes=1,n_samples=1000,cov=0.1,mean=[5.,2.,9.])
X4, Y4 = make_gaussian_quantiles(n_features=3, n_classes=1,n_samples=1000,cov=0.1,mean=[1.,2.,9.])
X5, Y5 = make_gaussian_quantiles(n_features=3, n_classes=1,n_samples=1000,cov=0.1,mean=[2.,9.,1.])

Y2=Y2+1
Y3=Y3+2
Y4=Y4+3
Y5=Y5+4

X=np.vstack((X1,X2,X3,X4,X5))
Y=np.hstack((Y1,Y2,Y3,Y4,Y5))

#%% Plotear los puntos y ver que den bien
fig=plt.figure()
ax=fig.gca(projection='3d')
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,s=75)
plt.title('Datos generados aleatoriamente para K-Means')

fig=plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,s=120)
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('Datos generados aleatoriamente para K-Means Original')
#%%

def KMEANS(k,y,datos):
    yp=np.zeros(y.shape[0])
    K_nuevos=np.zeros((k.shape[0],k.shape[1]))
    #   Lleno los valores de las clases a la que pertenece cada punto
    
    for i in range (datos.shape[0]):
#        norm= np.sqrt(np.sum((self.X - X[idx].reshape((1,self.X.shape[1])))**2,axis=1))

        norm= np.sqrt(np.sum((k - datos[i].reshape((1,datos.shape[1])))**2,axis=1))
        id_k=norm.argsort()[:k.shape[0]]    #guarda el orden de los k-means de menor a mayor en cuanto a lejania
        k_vect=y[id_k]       
        m = stats.mode(k_vect)
        yp[i]=m[0][0]    #toma el valor del k-mean que más se repite
                
    #   Calculo cual es el punto medio del grupo
    for jdx in range (k.shape[0]):
        yp[yp==jdx]=1
        yp[yp!=1]=0
        dat=datos.copy()
        dat=dat.T
        dat=(dat*y).T
        dat=dat.sum(axis=0)        
        dat=np.mean(dat,axis=0)
        K_nuevos[jdx]=dat
        
    return yp,K_nuevos               
#%%    
k_mean=5
k_original=np.random.randn(k_mean,dim)*10.0
k_original[k_original<1]*=(-1)

k_cambia=np.ones((k_original.shape[0],k_original.shape[1]))
k_final=np.zeros((k_original.shape[0],k_original.shape[1]))
aux=np.ones((k_original.shape[0],k_original.shape[1]))
Y_iguales=np.ones(Y.shape[0]).ravel()
yp,k_cambia = KMEANS(k_original,Y_iguales,X)
d=0
#np.mean((k_cambia-aux)**2)
while (np.mean((k_cambia-aux)**2)>0.005):
#while (np.linalg.norm((k_cambia,aux))>1):
        if(d==0):
            fig=plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=yp,s=120)
            plt.xlabel('X', size=14)
            plt.ylabel('Y', size=14)
            plt.title('K-Means Clasificados al Principio')
        print(d)
        print(np.mean((k_cambia-aux)**2))
    
    
        if(d==1):
            fig=plt.figure()
            plt.scatter(X[:, 0], X[:, 1], marker='o', c=yp,s=120)
            plt.xlabel('X', size=14)
            plt.ylabel('Y', size=14)
            plt.title('K-Means Clasificados al Principio')
        print(d)
        print(np.mean((k_cambia-aux)**2))
    
    
        d+=1
        aux=k_cambia
        yp,k_cambia = KMEANS(k_cambia,yp,X)
        
        
k_final=k_cambia
print(d)      







    