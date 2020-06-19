# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:30:57 2019

@author: TanPham
"""

import numpy as np
import matplotlib.pyplot as plt


def Cost(x):
    R = (1000 * (x[0] + x[1] + x[2]) + 750 * (x[3] + x[4] + x[5]) + 250 * (x[6] + x[7] + x[8]))
    # R= 0.6224*x[0]*x[2]*x[3]+ 1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+ 19.84*x[0]**2*x[2]
    return -R


def Constraint(x):
    g = np.zeros(9)
    geq = np.zeros(3)

    g[0] = x[0] + x[3] + x[6] - 400
    g[1] = x[1] + x[4] + x[7] - 600
    g[2] = x[2] + x[5] + x[8] - 300
    g[3] = 3 * x[0] + 2 * x[3] + x[6] - 600
    g[4] = 3 * x[1] + 2 * x[4] + x[7] - 800
    g[5] = 3 * x[2] + 2 * x[5] + x[8] - 375
    g[6] = x[0] + x[1] + x[2] - 600
    g[7] = x[3] + x[4] + x[5] - 500
    g[8] = x[6] + x[7] + x[8] - 325
    geq[0] = 3 * (x[0] + x[3] + x[6]) - 2 * (x[1] + x[4] + x[7])
    geq[1] = x[1] + x[4] + x[7] - 2 * (x[2] + x[5] + x[8])
    geq[2] = 4 * (x[2] + x[5] + x[8]) - 3 * (x[0] + x[3] + x[6])
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


    

class Particle:
    
    def __init__(self):
        self.Position=-1
        self.Velocity=-1
        self.Obj_val=0
        self.Personalbest_P=-1
        self.Personalbest_Value=-1
    
    def __repr__(self):
        return str(self.Position)

class Swarm:
    
    def __init__(self):
        self.Particle_list=[]
        self.Global_Best_Pos=[]
        self.Global_Best_Value=np.inf
    
    def Create_Swarm(self, no_P):
        for i in range(no_P):
            self.Particle_list.append(Particle())
        return self.Particle_list
    
    def Initialization(self,no_P):
            for i in range(no_P):
                self.Particle_list[i].Position=(ub-lb)*np.random.rand(dim)+lb
                self.Particle_list[i].Velocity=np.zeros(dim)
                self.Particle_list[i].Personalbest_P=np.zeros(dim)
                self.Particle_list[i].Personalbest_Value=np.inf
            self.Global_Best_Pos=np.zeros(dim)
            self.Global_Best_Value=np.inf
            return self.Particle_list, self.Global_Best_Pos, self.Global_Best_Value

def main():
    
    CC=np.zeros(maxIter)
    for i in range(maxIter):
        for k in range(noP):
            currentX=swarm.Particle_list[k].Position.copy()
            swarm.Particle_list[k].Obj_val=Fun(currentX)
            if swarm.Particle_list[k].Obj_val<swarm.Particle_list[k].Personalbest_Value:
                swarm.Particle_list[k].Personalbest_P=currentX.copy()
                swarm.Particle_list[k].Personalbest_Value=swarm.Particle_list[k].Obj_val
            if swarm.Particle_list[k].Obj_val<swarm.Global_Best_Value:
                swarm.Global_Best_Pos=currentX.copy()
                swarm.Global_Best_Value=swarm.Particle_list[k].Obj_val
        'Update'
        w=wMax-i*((wMax-wMin)/maxIter)
        
        for k in range(noP):
            swarm.Particle_list[k].Velocity=w*swarm.Particle_list[k].Velocity\
        + c1*np.random.rand(dim)*(swarm.Particle_list[k].Personalbest_P-swarm.Particle_list[k].Position)\
        + c2*np.random.rand(dim)*(swarm.Global_Best_Pos-swarm.Particle_list[k].Position)
            'Check velocity'
            index1 = swarm.Particle_list[k].Velocity > vMax
            index2 = swarm.Particle_list[k].Velocity < vMin
            swarm.Particle_list[k].Velocity[index1] = vMax[index1]
            swarm.Particle_list[k].Velocity[index2] = vMin[index2]
            'Update Position'
            swarm.Particle_list[k].Position += swarm.Particle_list[k].Velocity
            'Check position'
            index3 = swarm.Particle_list[k].Position > ub
            index4 = swarm.Particle_list[k].Position < lb
            swarm.Particle_list[k].Position[index3] = ub[index3]
            swarm.Particle_list[k].Position[index4] = lb[index4]
        
        CC[i]=swarm.Global_Best_Value
        print('Iteration:',i,'-Obj - ',swarm.Global_Best_Value,)
    print(swarm.Global_Best_Pos)
    plt.plot(CC)
    plt.show()
            

ub=np.array([300, 300, 300, 300, 300, 300, 300, 300, 300])
lb=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
dim=len(ub)
noP = 1000
maxIter = 3000
wMax = 0.9
wMin = 0.2
c1 = 2
c2 = 2
vMax = (ub - lb) * 0.2
vMin  = -vMax
swarm=Swarm()
swarm.Create_Swarm(noP)
swarm.Initialization(noP)
#print(s.Particle_list,s.Global_Best_Pos,s.Global_Best_Value)
#particle=Particle()
main()
