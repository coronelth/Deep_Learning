import numpy as np
import matplotlib.pyplot as plt

#%%----Elección de los parámetros

random_state = 159
np.random.seed(random_state)

muestras=5000
dim=5
A=np.random.rand(1,dim).ravel()*10


#%%-----Generación de los datos
amplitud=100
x=np.ones((muestras,1))
aux=amplitud*(np.random.rand(muestras,dim-1)*2+1)
x=np.append(x,aux,axis=1)
ruido=15 # amplitud ruido

y=x.dot((A.T))
y_ruido=(2*np.random.rand(muestras)-1)*ruido
for i in range (y.shape[0]):
    y[i]=y[i]+y_ruido[i]
    
def minimos_cuadrados(A,y):
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)

A_calc=minimos_cuadrados(x,y)

print("Parámetros reales:\t",A,"\n Parámetros calculados \t",A_calc,"\n")

x_label = (x[:,3])
func = y
#%%
error=np.mean((A-A_calc)**2)*100
print("Error en el cálculo de los parámetros:\t",error," % \n")
#%%
plt.scatter(x_label, func,edgecolors="red")
#plt.plot(np.linspace(100,300,50),A_calc[3]+A_calc[0]*np.linspace(100,300,50))
plt.xlabel('x',fontsize=14)
plt.ylabel('y(x)',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Datos generados proyectados en la cuarta dimensión para N=5000')
plt.grid(True)
plt.savefig("TP_1_Ej_1_regresion_lineal.png")
plt.show()
