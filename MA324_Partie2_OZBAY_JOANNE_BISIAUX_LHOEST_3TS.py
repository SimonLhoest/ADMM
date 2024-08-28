# -*- coding: utf-8 -*-
"""
JOANNE Nils
OZBAY Yasin
BISIAUX Benoit
LHOEST Simon

BE MA324 Partie 2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Chargement des données
iris=datasets.load_iris()
#on ne prend que les deux premières colonnes d'iris.
X, y = iris.data[:, :2], iris.target #On conserve 50% du jeu de données pour l'évalutation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#On prend que deux catégories
u_train=X_train[y_train==0,:]
v_train=X_train[y_train==1,:]
u_test=X_test[y_test==0,:]
v_test=X_test[y_test==1,:]

#%% IRIS

def init(u,v,n=2):
    '''
    Initialisation des paramètres A, C, Z, W et lambda de la fonction
    '''
    p=np.shape(u)[0]
    q=np.shape(v)[0]
    
    #Construction de C
    X=np.concatenate((-1*u, v),axis=0)
    e=np.block([[np.ones((p,1))], [-np.ones((q,1))]])
    C=np.block([X,e])
    #Construction de A
    A=np.eye(n+1)
    A[-1,-1]=0
    
    W=np.ones((n+1,1))
    Z=np.ones((p+q,1))
    lambo=np.ones((p+q,1))
    return W,Z,lambo,C,A

def Lpw(z,lambo,C,A,rho):
    '''
    Dérivée partielle du lagrangien augmenté en x
    '''
    ppq=np.shape(z)[0]
    un = (A+rho*(C.T@C))
    deux = -C.T@(rho*(z-np.ones((ppq,1)))+lambo) #d=-np.ones
    #LL=1/2*np.linalg.norm(w)**2 + lambo.T@(C@np.concatenate((w,b),axis=0))+z+np.ones((p+q),1) + rho/2*np.linalg.norm(C@np.concatenate((w,b),axis=0)+z+np.ones((p+q),1))**2
    L= np.linalg.inv(un)@deux
    return  L

def Lpz(w,lambo,C,A,rho):
    '''
    Dérivée partielle du lagrangien augmenté en z
    '''
    ppq=C.shape[0]
    z=-lambo/rho-C@w + np.ones((ppq,1))
    return z
    
def proj(x):
    '''
    Projection dans R+
    '''
    y=np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i]=np.maximum(0,x[i])
        
    return y
    
def calculw(w,z,lambo,eps,C,A,rho):
    '''
    Algorithme P
    '''
    ppq=np.shape(z)[0]
    w1=Lpw(z,lambo,C,A,rho)
    z1=proj(Lpz(w1,lambo,C,A,rho))
    lamb1=lambo+rho*(C@w1+z1+np.ones((ppq,1)))
    it=1
    while np.linalg.norm(w1-w)>eps :
        w=w1
        z=z1
        lambo=lamb1
        w1=Lpw(z,lambo,C,A,rho)
        z1=proj(Lpz(w1,lambo,C,A,rho))
        lamb1=lambo+rho*(C@w1+z1+np.ones((ppq,1)))
        it+=1
    return w,it


def affichage(u,v,w):
    '''
    Affichage des données avec les droites
    '''
    w=w[:len(w)-1,0]    
    b=(np.min(w @ u.T)+np.max(w @ v.T))/2
    delta=(np.min(w@ u.T)-np.max(w@ v.T))/2 
    #Affichage des points    
    plt.scatter(u[:,0], u[:,1],color='r')
    plt.scatter(v[:,0], v[:,1], color='b')
    #Calcul droites
    x = np.linspace(3,8,50)
    y=(-w[0]*x  + b)/w[1]
    y1=(-w[0]*x + b+delta)/w[1]
    y2=(-w[0]*x  + b-delta)/w[1]
    #Affichage droites
    plt.grid()
    plt.plot(x,y,color='g' )
    plt.plot(x, y1,color='y' )
    plt.plot(x,y2,color='m')

def ADMM(u,v,rho=0.01,eps=10**(-2),n=2):
    W,Z,lambo,C,A=init(u,v,n)
    w,it=calculw(W,Z,lambo,eps,C,A,rho)
    return w,it

w,it=ADMM(u_train,v_train)
print("Nombre d'itération(s) : ",it)
affichage(u_train,v_train,w)
plt.title("Algoritme ADMM sur les données IRIS")

