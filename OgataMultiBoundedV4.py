#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:44:24 2021

@author: phicuong
"""



import math
import random
import numpy as np
from base_functions import *
import time

##Algo 1: Ogata for BOUNDED in high dimension
##time interval [TStart, TEnd]  ##to simplify [0,TEnd]

## Linear Hawkes process phi_i,t = (mu_i + \sum_j int_{0}^t h_ji(t - s) dZ^j_s) 1_{a_i(t) > delta}


##Initial a list of Points with 3 row and 0 column
##the first row for AllId, the second for time and last row for thinning mark



# PARAMETERS #######################################################################################################################################################################

print ('SET PARAMETERS')

#np.random.seed(1)


NbId = 101
AllId = range(NbId)  ##0,1,2,...9, 10
alpha = 2
delta           = 0.01
a               = 0.1
TStart = 0
TEnd = 10


# Controle global weight of connections
Co      = 2


Points = np.empty((3,0))
aldel = alpha * delta  

scale = 1
mu = scale * np.ones((NbId, 1))
geo_sum = (1- np.exp(-aldel*(math.floor(a/delta)+1)))/((1- np.exp(-aldel)))



omega           = np.zeros((NbId, NbId))
sum_omega = np.zeros(NbId)
for i in AllId:
    # Synaptic weights
    # print ('NEURON N°'+ str(i+1) +' Compute : Interaction function / Kalikow decompostion / Majorant')
    omega[i,i] = Co
    id_set = list(range(NbId))
    del id_set[i]
    for j in id_set:
        omega[j,i]  = Co / (2 * (abs(j-i)**6)) 
    sum_omega[i] = np.sum(omega[:,i])


##The dominating intensities M_i

Majorante = np.zeros(NbId)

for i in AllId :
    Majorante[i] = mu[i]+ sum_omega[i]*geo_sum
    



# calcul the intensity function of AllId i, TIME t, LAST SPIKE time is lastspike 

def FuncPhi(i, t, Points, alpha, a, delta):
    res = 0
    if (t - ListSpike[i][-1] > delta): 
        IDprev_points = np.where((Points[1, :] >= t-a) & (Points[1, :] < t) & (Points[2,:] == 1))  ##return an array of 2 dimensions, 
        lpoints = IDprev_points[0]  ##[0]: column, [1]: row
        #print(lpoints)
        for idp in lpoints:
            idpoints2 = Points[0, idp]
            tpoints2 = Points[1,idp]
            res = res + omega[idpoints2,i]*np.exp(-alpha*(t- tpoints2))
        res = mu[i]+ res
    return(res)
    
            



###Init points for thinning
print("Init points for thinning")

for i in AllId:
    print(i)
    tnext = TStart + rexp(1, Majorante[i])
    while tnext < TEnd :
        #print(tnext)
        newpoint = [i, tnext, -1]
        Points = np.c_[Points, newpoint]
        tnext = tnext + rexp(1, Majorante[i])
    #endwhile
#endfor



ListIDsorted = Points[1,:].argsort()   ##sorting Points by time 


##Forward thinning
print("Forward thinning")

tic = time.time()
ListSpike = [[-delta] for i in range(NbId)] 

for idx in ListIDsorted:
    pts = Points[:,idx]
    #print(pts)
    i = pts[0]
    t = pts[1]
    u = runif(1, 0, 1)
    #print(u)
    phi= FuncPhi(i, t, Points, alpha, a, delta)
    
    #print("phi",phi)
    if u < phi/Majorante[i]:
        Points[2,idx] = 1
        ListSpike[i].append(t)
        #print("I accept a point",lastspike)
        #print("I accept a point with phi", phi)
    else:
        Points[2,idx] = 0
    #print(DerniereSpike)
    #print("ListIntLast")
    #print(ListIntLast)
toc = time.time()
        
PointsAccepted = Points[:,Points[2,]== 1]

print("Total time needed", toc - tic)
np.savetxt('/Users/phicuong/Downloads/Kalikow/test.csv', PointsAccepted, delimiter = ',')
