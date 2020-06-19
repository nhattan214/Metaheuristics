# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:30:57 2019

@author: TanPham
"""

import numpy as np
import matplotlib.pyplot as plt

def Obj(X):
    result=sum(i**2 for i in X)+1
    return result



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
    
    CC=np.zeros(maxIter+1)
    for i in range(maxIter):
        for k in range(noP):
            currentX=swarm.Particle_list[k].Position.copy()
            swarm.Particle_list[k].Obj_val=Obj(currentX)
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
            index1=swarm.Particle_list[k].Velocity>vMax
            index2=swarm.Particle_list[k].Velocity<vMin
            swarm.Particle_list[k].Velocity[index1]=vMax[index1]
            swarm.Particle_list[k].Velocity[index2]=vMin[index2]
            swarm.Particle_list[k].Position+=swarm.Particle_list[k].Velocity
            'Check position'
            index3=swarm.Particle_list[k].Position>ub
            index4=swarm.Particle_list[k].Position<lb
            swarm.Particle_list[k].Position[index3]=ub[index3]
            swarm.Particle_list[k].Position[index4]=lb[index4]
        
        CC[i]=swarm.Global_Best_Value
        print('Iteration:',i,'-Obj - ',swarm.Global_Best_Value)
            

ub = np.array([10, 10, 10, 10, 10, 10 ,10, 10, 10, 10])
dim=len(ub)
lb = np.array([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10])
noP = 30
maxIter = 500
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
