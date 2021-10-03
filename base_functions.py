# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from random import random
import numpy as np
import matplotlib.pyplot as plt


from scipy.stats import uniform
from scipy.stats import expon
from scipy.stats import poisson


def runif(N, a, b):  ##Generate N rv in [a,b]
    return(uniform.rvs(loc = a, scale = b-a, size = N))


def rexp(N,lbda):   ##Exponential rv of parameter lambda
    return(expon.rvs(scale = 1/lbda, size =N))


def rpois(N, mu):
    return(poisson.rvs(mu, size =N)) 
    


def sim_pois(Tstart,Tend,M):
  N = rpois(1, M*(Tend-Tstart))
  x = runif(N,Tstart,Tend)
  return(sorted(x))
