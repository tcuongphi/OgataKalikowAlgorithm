#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 22:56:57 2021

@author: phicuong
"""

import math
import numpy as np

from Kalikow_base_function import *

# CLASS #######################################################################################################################################################################

class Neuron:

    # Neuron paramters 
    i               = 0
    
    
    #Initial points of Poisson process
    init_time = np.empty(0)
    nb_init_point = 0
    

    
    def __init__(self, i, nb_id, mu, delta, alpha, a, al_del, geo_sum, omega, maj):
        
        # Neuron index
        self.i = int(i)
        # Number of pre-synaptic neurons
        self.nb_syn = nb_id
        #spontaneous rate
        self.mu = mu
        # refractory period
        self.delta = delta
        # coefficient of interaction function
        self.alpha = alpha
        self.a = a
        self.al_del = al_del
        self.geo_sum = geo_sum
        self.omega = omega
        # Initial intensity
        self.maj = maj

    def initiate_point(self,start,end):

        # Poisson process with the initial intensity
        self.init_time          = PoissonProcess(start,end,self.maj)
        #print(self.init_time)
        self.nb_init_point      = self.init_time.shape[0]
    
    def get_init_point(self):
        return(self.init_time)

    
    def thinning(self, id_point, times, index, init_selected, last_spike):
        cur_time = times[id_point]
        cur_neur = int(index[id_point])
        out = 0       
        #CHECK REFRACTORY PERIOD####
        intensity = 0
        if cur_time - last_spike[cur_neur] > self.delta :
            prev_points = Point_in_Interval(times, cur_time, self.a)
            pts_time = 0
            point_neur= 0

            ##Compute the intensity
            for ind in prev_points:
                if init_selected[ind] == 1:
                    pts_time = times[ind]
                    point_neur = int(index[ind])
                    intensity = intensity + np.exp(-self.alpha*(cur_time - pts_time)) *self.omega[point_neur]
            intensity = intensity + self.mu
        
        
        # ##Forward thinning
        thinning_proba  = 0
        init_unif        = 0
        thinning_proba = intensity / self.maj
        # Pick a uniform random variable for each initial point
        init_unif          = RandomUnif(0,1,1)
        # Determine for this point has to be kept
        out      = 1 if init_unif < thinning_proba else -1    
        return(out)
        
    def set_final_spike_train(self, init_selected):
        lst = np.where(int_selected==1)[0]
        self.neur_spike = index[lst]
        self.time_spike = times[lst]
    #def talk(self):
    #    print("list", id_prev_points)
    #    print("My final spike train", self.sel_time)

