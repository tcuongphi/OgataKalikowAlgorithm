#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 23:06:03 2021

@author: phicuong
"""



from Ogata_neuron import Neuron

import math
import numpy as np

import time
# PARAMETERS #######################################################################################################################################################################

# Time parameters
TStart  = 0
TEnd    = 10

# Number of neurons
NbId    = 101
AllId   = list(range(NbId))

# Controle global weight of connections
Co      = 2

##############################################################################################################################################################################

# NEURONS INSTANTIATION #######################################################################################################################################################################

omega           = np.zeros((NbId, NbId))
#res             = np.zeros((NbId, NbId))
maj             = np.zeros(NbId)

mu              = 1
alpha           = 2
delta           = 0.01
a               = 0.1

al_del          = alpha * delta
geo_sum = (1- np.exp(-al_del*(math.floor(a/delta)+1)))/((1- np.exp(-al_del)))


tic = time.time()

for i in AllId:
    # Synaptic weights
    print ('NEURON N°'+ str(i+1) +' Compute : Interaction function  / Majorant')
    omega[i,i] = Co
    id_set = list(range(NbId))
    del id_set[i]
    for j in id_set:
        omega[i,j]  = Co / (2 * (abs(j-i)**6)) 
    maj[i] = mu+ np.sum(omega[:,i])*geo_sum    

Neur       = np.empty((NbId),Neuron)
for i in AllId:
    print ('NEURON N°'+ str(i+1) +' INSTANTIATION')
    Neur[i] = Neuron(i, NbId, mu, delta, alpha, a, al_del, geo_sum, omega[i], maj[i])





##############################################################################################################################################################################

# SPIKE TRAINS SIMULATION AND PRESELECTION #######################################################################################################################################################################

# Run parallel preselection
for i in AllId:
    Neur[i].initiate_point(TStart,TEnd)

# Get and aggregate togather the whole set of spikes trains.
times                    = np.empty((0))
index                   = np.empty((0))
last_spike              = np.zeros(NbId)


def update_last_spike(id_point, temps):
    neur_point = index[id_point]
    last_spike[int(neur_point)] = temps

for i in AllId:
    neur_time = Neur[i].get_init_point()
    times = np.hstack((times, neur_time))
    nb_points = neur_time.shape[0]
    neur_id = i*np.ones(nb_points)
    index = np.hstack((index, neur_id))
    
#Sorting points
order = times.argsort()
times = times[order]
index = index[order]
total_nb_points = len(times)
init_selected   = np.zeros(total_nb_points) 
#Forward thinning
for id_point in range(total_nb_points):
    init_selected[id_point] = Neur[int(index[id_point])].thinning(id_point, times, index, init_selected, last_spike) 
    if init_selected[id_point] == 1:
        update_last_spike(id_point, times[id_point])


lst = np.where(init_selected == 1)[0]
time_final = times[lst]
neur_final = index[lst]
spike_train = np.vstack((time_final, neur_final))
toc = time.time()
print(toc - tic)

np.savetxt("spike_train_"+str(NbId)+".csv", spike_train, delimiter=",")
