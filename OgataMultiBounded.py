#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:39:38 2021

@author: phicuong
"""

import math
from base_functions import *


##Algo 1: Ogata for BOUNDED in high dimension
##time interval [tmin, tmax]  ##to simplify [0,tmax]

## Linear Hawkes process phi(t) = mu + int_{0}^t h(t - s) dZ_s 


##Initial a list of Points with 3 row and 0 column
##the first row for index, the second for time and last row for thinning mark


nbId = 5
INDEX = range(0,nbId)  ##0,1,2,...9
alpha = 2
delta = 0.7
tmin = 0
tmax = 5





Points = np.empty((3,0))
aldel = alpha * delta  

scale = 0.4
mu = scale * np.ones((nbId, 1))


p = 5
Lambda = np.zeros((nbId, nbId))
for i in INDEX :
    for j in INDEX :
        if (j != i):
            Lambda[j,i] = 1/(4*(abs(j-i)**p))
    Lambda[i,i] = 1- np.sum(Lambda[:, i])

##ATTENTION: We don't do Kalikow here, but we need Lambda here fore the calculation of beta. Since we want to make comparision later,
## that why we keep beta[i,i] = Lambda[i,i], then Lambda need to be defined.

beta = np.zeros((nbId, nbId))
sum_beta = np.zeros(nbId)
for i in INDEX :
    for j in INDEX :
        if (j != i):
            beta[j,i] = 1/(2*(abs(j-i)**6))
    beta[i,i] = Lambda[i,i]
    sum_beta[i] = np.sum(beta[:,i])  ##\sum_{j} beta_ji



##The dominating intensities M_i

Majorante = np.zeros(nbId)

for i in INDEX :
    Majorante[i] = mu[i]+ sum_beta[i]/((1- np.exp(-aldel)))
    



## calcul the intensity function of INDEX i, TIME t, LAST SPIKE time is lastspike 

def FuncPhi(i,t, Pts, lastspike, alpha, delta):
    res = 0
    if t - lastspike > delta:
        res = mu[i]
        listIDpoints = np.where((Pts[1, :] < t) & (Pts[2,:] == 1))  ##return an array of 2 dimensions, 
        lpoints = listIDpoints[0]  ##[0]: column, [1]: row
        #print(lpoints)
        for idp in lpoints:
            idpoints = Pts[0, idp]
            tpoints = Pts[1,idp]
            res = res + beta[idpoints,i]*np.exp(-alpha*(t- tpoints))
    return(res)
    
            





###Init points for thinning


for i in INDEX:
    tnext = tmin + rexp(1, Majorante[i])
    while tnext < tmax :
        newpoint = [i, tnext, -1]
        Points = np.c_[Points, newpoint]
        tnext = tnext + rexp(1, Majorante[i])
    #endwhile
#endfor



ListIDsorted = Points[1,:].argsort()   ##sorting Points by time 


##Forward thinning

DerniereSpike = 0

for idx in ListIDsorted:
    pts = Points[:,idx]
    idpts = pts[0]
    temps = pts[1]
    u = runif(1, 0, 1)
    if u < FuncPhi(idpts, temps, Points, DerniereSpike, alpha, delta)/Majorante[idpts]:
        Points[2,idx] = 1
        DerniereSpike = temps
    else:
        Points[2,idx] = 0


