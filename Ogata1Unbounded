#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:42:02 2021

@author: phicuong
"""

import math
from base_functions import *


##Algo 1: Ogata for UNBOUNDED in 1 dimension
##time interval [tmin, tmax]  ##to simplify [0,tmax]


## Linear Hawkes process phi(t) = mu + int_{0}^t h(t - s) dZ_s 

mu = 0.3

##the function phi with h(t)= beta* exp(-alpha*t)
def FuncPhi(t, Pts):
    #lpoints = [x for x in Pts if x < t]
    lpoints = Pts[Pts < t]
    #z =list(map(math.exp,-t + lpoints))   ##We can do more efficient here by keeping the value at the last accepted point
    #return(mu + sum(z))
    z = np.exp(-t + lpoints)
    return(mu + z.sum())

Gamma = mu 
s = 0 ##tmin
tmax = 10
alpha = 1 ## h(0) = exp(0)  ## the jump size of the intensity 

Points = np.empty((1,0))


##Init the first point

s = s + rexp(1, Gamma)

while s < tmax :
    u = runif(1, 0 ,1)
    #print(FuncPhi(s, Points))
    if u< FuncPhi(s, Points)/Gamma :
        #Points.append(s)    # Points = np.append(Points, s)  ##np.c_[Points, s] column
        Points = np.append(Points, s)
        break
    s = s + rexp(1, Gamma)
    #print(s)

##attention at the case no point in Points after this while loop (general)
##in LHP, we always accept the first simulated point

##Init a flag to skip the first line after the while loop
Flag = True
s = s + rexp(1, Gamma)


while s < tmax:
    if Flag == True:
        lastpoint = Points[-1]
        Gamma = FuncPhi(lastpoint, Points) + alpha   ##alpha is the jump size of the intensity 
    v = runif(1, 0, 1)
    aval = FuncPhi(s, Points)
    if v < aval/Gamma :
        #Points.append(s)
        Points = np.append(Points, s)
        Flag = True
    else:
        Flag = False
        Gamma = FuncPhi(s, Points)
    s = s + rexp(1, Gamma)    

print(Points)
