# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:15:12 2019

@author: May33
"""
import copy
import numpy as np
import random
import matplotlib.pyplot as plt

def initialization(num_searchagent, Ub, Lb):
    Positions=np.zeros((num_searchagent, len(Ub)))
    dim=len(Lb);
    for i in range(num_searchagent):
        for j in range(dim):
            Positions[i][j]=(np.random.uniform(low=Lb[j],high=Ub[j]))
    return Positions

def Cost(x):
    R= 0.6224*x[0]*x[2]*x[3]+ 1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+ 19.84*x[0]**2*x[2]
    return -R

def Constraint(x):
    g=np.zeros(len(x))
    g[0]=-x[0]+0.0193*x[2]
    g[1]=-x[1]+0.00954*x[2]
    g[2]=-(np.pi)*x[2]**2*x[3]-(4/3)*np.pi*x[2]**3+1296000
    g[3]=x[3]-240
    geq=[]
    return g, geq

def geteqH(g):
    if g==0:
        H=0
    else:
        H=1
    return H

def getH(g):
    if g<=0:
        H=0
    else:
        H=1
    return H

def getConstraint(x):
    lam=10**15
    lameq=10**15
    Z=0
    g, geq= Constraint(x)
    for k in range(len(g)):
        Z+=lam*g[k]**2*getH(g[k])
    for j in range(len(geq)):
        Z+=lameq*geq[j]**2*geteqH(geq[j])
    return Z

def Fun( x):
    Z=Cost(x)
    Z+=getConstraint(x)
    return Z

def GWO(SearchAgents_no,Max_iter,ub,lb,dim):
    
    Alpha_pos=np.zeros(dim)
    Alpha_score=np.inf
    
    Beta_pos=np.zeros(dim)
    Beta_score=np.inf
    
    Delta_pos=np.zeros(dim)
    Delta_score=np.inf
    
    Positions=initialization(SearchAgents_no,ub,lb)
	
    Convergence_curve=np.zeros(Max_iter+1)
    velocity=0.3*np.random.rand(SearchAgents_no,dim)
    w=0.5+np.random.rand()/2
    l=0
    while l<Max_iter:
        for i in range(0,SearchAgents_no):
            Flag4ub=Positions[i]>ub
            Flag4lb=Positions[i]<lb
            Positions[i]=(Positions[i]*(~(Flag4ub+Flag4lb)))+ub*Flag4ub+lb*Flag4lb
#            print(Positions[i])
            fitness=Fun(Positions[i])
            if fitness<Alpha_score:
                Alpha_score=fitness
                Alpha_pos=Positions[i].copy()
                
            if ((fitness>Alpha_score) and (fitness<Beta_score)):
                Beta_score=fitness
                Beta_pos=Positions[i].copy()
                
            if (fitness>Alpha_score) and (fitness>Beta_score) and (fitness<Delta_score):
                Delta_score=fitness
                Delta_pos=Positions[i].copy()
                
        a=2-l*((2)/Max_iter)
        
        for i in range(0,SearchAgents_no):
            for j in range(len(Positions[0])):
                r1=random.random()
                r2=random.random()
                
                A1=2*a*r1-a
#                C1=2*r2
                C1=0.5
                
                D_alpha=abs(C1*Alpha_pos[j]-w*Positions[i][j])
                X1=Alpha_pos[j]-A1*D_alpha
                
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a
#                C2=2*r2
                C2=0.5
                
                D_beta=abs(C2*Beta_pos[j]-w*Positions[i][j])
                X2=Beta_pos[j]-A2*D_beta
                
                r1=random.random()
                r2=random.random()
                r3=random.random()
               
                A3=2*a*r1-a
#                C3=2*r2
                C3=0.5
                
                D_delta=abs(C3*Delta_pos[j]-w*Positions[i][j])
                X3=Delta_pos[j]-A3*D_delta
                
                velocity[i][j]=w*(velocity[i][j]+C1*r1*(X1-Positions[i][j])+C2*r2*(X2-Positions[i][j])+C3*r3*(X3-Positions[i][j]))
                Positions[i][j]=Positions[i][j]+velocity[i][j]
        l+=1
     
        Convergence_curve[l]=Alpha_score

    return Alpha_score, Alpha_pos, Convergence_curve
    
SearchAgents_no=50
Max_iter=3000
Ub=np.array([99, 99, 200, 200])
Lb=np.array([0, 0, 10, 10])
dim=len(Lb)


Best_score, Best_pos, CC=GWO(SearchAgents_no,Max_iter,Ub,Lb,dim)
print(Cost(Best_pos))
plt.plot(CC)
plt.xlabel('Iteration')
plt.ylabel('Obj Value')
plt.title('Convergence rate ' +str(Best_score))
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.show()
