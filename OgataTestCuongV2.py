#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:15:58 2021

@author: phicuong
"""

import math
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf



# INTEGRAL OF THE INTENSITY (RESCALING TIME) #######################################################################################################################################################################

def rescale_time(cur_neur, alpha, a, mu, prev_neur, prev_time, left, right):

    out = 0 

    nb_interval = left.shape[0]
    # integrate the mu intensty
    for b in range(nb_interval):
        out = out + mu * (right[b] - left[b])

    # integrate the interactive intensty
    nb_spikes = prev_neur.shape[0]
    for s in range(nb_spikes):

        spike_neur = prev_neur[s]
        spike_time = prev_time[s]
        for b in range(nb_interval):
            # Compute the boundary for the interactive intensity
            left_b  = max(spike_time, min(left[b] , spike_time + a))
            right_b = max(spike_time, min(right[b], spike_time + a))
            # Integrate the interaction function in [left_b;right_b]
            if(right_b > left_b):
                out = out + omega[int(spike_neur),int(cur_neur)] * math.exp(alpha * spike_time) * (math.exp(- alpha * left_b) - math.exp(- alpha * right_b)) / alpha

                #print(cur_neur)
                #print(spike_neur)
                #print(omega[int(cur_neur),int(spike_neur)])

    return(out)


# PARAMETERS #######################################################################################################################################################################

print ('SET PARAMETERS')

# Time parameters
TStart      = 0
TEnd        = 10

# Number of neurons
NbId        = 101
AllId       = range(NbId)

# Controle global weight of connections
#Co              = 2

omega           = np.zeros((NbId, NbId))
lamb            = np.zeros((NbId, NbId + 1))
maj_neigh       = np.zeros((NbId, NbId + 1))

mu              = 1
alpha           = 2
delta           = 0.01
a               = 0.1
Co = 2
# p = 5
# Lambda = np.zeros((NbId, NbId))
# for i in AllId :
#     for j in AllId :
#         if (j != i):
#             Lambda[j,i] = 1/(4*(abs(j-i)**p))
#     Lambda[i,i] = 1- np.sum(Lambda[:, i])

# ##ATTENTION: We don't do Kalikow here, but we need Lambda here fore the calculation of beta. Since we want to make comparision later,
# ## that why we keep beta[i,i] = Lambda[i,i], then Lambda need to be defined.

# beta = np.zeros((NbId, NbId))
# sum_beta = np.zeros(NbId)
# for i in AllId :
#     for j in AllId :
#         if (j != i):
#             beta[j,i] = 1/(2*(abs(j-i)**6))
#     beta[i,i] = Lambda[i,i]
#     sum_beta[i] = np.sum(beta[:,i])  ##\sum_{j} beta_ji




omega           = np.zeros((NbId, NbId))
sum_omega = np.zeros(NbId)
for i in AllId:
    # Synaptic weights
    print ('NEURON N°'+ str(i+1) +' Compute : Interaction function / Kalikow decompostion / Majorant')
    omega[i,i] = Co
    id_set = list(range(NbId))
    del id_set[i]
    for j in id_set:
        omega[j,i]  = Co / (2 * (abs(j-i)**6)) 
    sum_omega[i] = np.sum(omega[:,i])



#al_del          = alpha * delta
# geom_sum        = (1 - math.exp(- alpha * a)) / (1 - math.exp(- al_del))

# for i in AllId:
#     # Synaptic weights
#     omega[i,i] = Co
#     id_set = list(range(NbId))
#     del id_set[i]
#     for j in id_set:
#         omega[i,j]  = Co / (2 * (abs(j-i)**6)) 

#     # Kalikow decomposition
#     den = mu + geom_sum * sum(omega[i])
#     lamb[i,NbId] = mu / den
#     for j in AllId:
#         lamb[i,j] = geom_sum * omega[i,j] / den

#     # Intensity depending on the neighborhood 
#     maj_neigh[i, NbId] = mu / lamb[i, NbId]
#     for j in AllId:
#         maj_neigh[i,j] = omega[i,j] * geom_sum / lamb[i,j]

# IMPORT DATA #######################################################################################################################################################################

print ('IMPORT DATA')

spike_train         = np.loadtxt('spike_train_101.csv', delimiter=',', unpack=True)
time                = spike_train[:,0]
neur                = spike_train[:,1]
nb_spikes           = time.shape[0]

spike_neur          = list()
time_neur           = list()
nb_spikes_neur      = np.zeros((NbId),int)
for i in AllId:
    spike_neur.append(np.where(neur == i)[0])
    time_neur.append(time[spike_neur[i]]) 
    nb_spikes_neur[i] = time_neur[i].shape[0]  

# INTEGRATION BOUNDARIES #######################################################################################################################################################################

print ('DEFINES INTEGRATION BOUDARIES')

left                = list()
right               = list()

for i in AllId:
    left.append(TStart * np.ones((1)))
    right.append(np.empty((0)))
    nb_left = 1
    nb_right = 0
    for s in range(nb_spikes_neur[i]):
        print(s)
        # if the spike occurs near the end the right bound is pushed on left
        if(time_neur[i][s]>TEnd-delta):
            right[i] = np.hstack((right[i],time_neur[i][s]))
            nb_right = nb_right + 1
        # Otherwise a new interval is created
        else:
            left[i]     = np.hstack((left[i],time_neur[i][s]+delta))
            right[i]    = np.hstack((right[i],time_neur[i][s]))
            nb_left     = nb_left + 1
            nb_right    = nb_right + 1
        print(left[i])
        print(right[i])
    # Close last intervall with TEnd if it is necessary
    if(nb_left>nb_right): 
        right[i]        = np.hstack((right[i],TEnd))
    print(i)
    print(left[i])
    print(right[i])
    print(left[i][left[i]>right[i]])




# TIME RESCALING #######################################################################################################################################################################

print ('RESCALES TIMES')

time_rescaled           = np.zeros((nb_spikes))

for s in range(nb_spikes):
#for s in range(10):
    print("spike " + str(s+1) + "/" + str(nb_spikes))
    cur_neur            = int(neur[s])
    cur_time            = time[s]
    prev_neur           = neur[time<cur_time]
    prev_time           = time[time<cur_time]
    prev_left           = left[cur_neur][left[cur_neur]<time[s]]
    prev_right          = right[cur_neur][left[cur_neur]<=time[s]]
    time_rescaled[s]    = rescale_time(cur_neur, alpha, a, mu, prev_neur, prev_time, prev_left, prev_right)

time_rescaled_neur  = list()
isi_rescaled_neur   = list()
for i in AllId:
    time_rescaled_neur.append(time_rescaled[neur==i])
    isi_rescaled_neur.append(list())
    for s in range(nb_spikes_neur[i]-1):
        isi_rescaled_neur[i].append(time_rescaled_neur[i][s+1]-time_rescaled_neur[i][s])


p_value = np.zeros((NbId))
p_value2 = np.zeros((NbId))
for i in AllId:
    print(i)
    print(time_rescaled_neur[i])
    print(isi_rescaled_neur[i])
    if len(isi_rescaled_neur[i]) >0:
        ks_test =  kstest(isi_rescaled_neur[i], "expon")
        p_value[i] = ks_test[1]
        ks_test2 = kstest(time_rescaled_neur[i], "uniform", args=(0, time_rescaled_neur[i][-1]) )
        p_value2[i] = ks_test2[1]
        #plot_acf(isi_rescaled_neur[i], lags=10)
        #plt.rc("figure", figsize=(20,10))
        #plt.figure(figsize=(20,10))
        #plot_acf(isi_rescaled_neur[i], lags = len(isi_rescaled_neur[i])-1)
        



order = p_value.argsort()   
p_value = p_value[order]

order2 = p_value2.argsort()
p_value2 = p_value2[order2]

print(p_value)
print(p_value2)

x = np.linspace(0, 1, NbId)

plt.plot([0.0, 1.0], [0.0, 1.0], 'r-', lw=2, label = "Targeted cdf (Uniform law)")
plt.scatter(x,p_value, label = "Empirical cdf")

plt.title('Cdf des p_values du Ks test.png')
plt.xlabel('X')
plt.ylabel('P(pvalue < X)')

plt.legend()

plt.savefig('Cdf des p_values du Ks test.png')
plt.show()




