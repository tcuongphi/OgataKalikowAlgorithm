#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:20:39 2021

@author: phicuong
"""
import math
import random
import numpy as np
from base_functions import *
import sys


###Ogata-Kalikow algorithm for LHP with hard refractory period


##Here we consider LINEAR HAWKES PROCESS with HARD REFRACTORY PERIOD
##input: mu_i, h (h_ji(t) = beta_ji exp(-alpha t)), delta( length of refractory period), 
##       tstart, tend
##We consider I = 1, 2, ..., NbId the set of indices

##init  ##to simplify mu_i = mu
##Here we choose very simple form of neighborhood v_j={j}x [0,t)
##beta_ji = beta_i(j) = 1/ (2 |j-i|^6) and beta_i = Constant

##weights lamb_ji = 1/4 |j-i|^4 and lamb_i = 1- sum_j lamb_ji


##Initial value
np.random.seed(1)

NbId = 101
AllId = range(NbId)

mu = 1
alpha = 2
delta = 0.01
a = 0.1

aldel = alpha*delta

tstart = 0
tend = 10
Co = 2

##all the functions necessary

lamb = np.zeros((NbId, NbId))
for i in AllId :
  for j in AllId :
    if (j != i) :
      lamb[i,j] = 1/(4*abs(j-i)**4)
      #print(lamb[i,j])
  lamb[i,i] = 1- sum(lamb[i,])  

##The dominating intensity M_i
domi = 1 - np.exp(-aldel*(np.floor(a/delta)+1))

majorante = np.zeros(NbId) 
for i in AllId:
    majorante[i] = max(4*domi/(1- np.exp(-aldel)), mu/lamb[i,i] + Co*domi /(lamb[i,i]*(1- np.exp(-aldel))))



omega           = np.zeros((NbId, NbId))
sum_omega = np.zeros(NbId)
for i in AllId:
    # Synaptic weights
    #print ('NEURON N°'+ str(i+1) +' Compute : Interaction function / Kalikow decompostion / Majorant')
    omega[i,i] = Co
    id_set = list(range(NbId))
    del id_set[i]
    for j in id_set:
        omega[j,i]  = Co / (2 * (abs(j-i)**6)) 
    sum_omega[i] = np.sum(omega[:,i])



##interaction function h definir par beta, alpha ou h_ji(t) = beta_ji exp(-alpha t)
def interact(t, i, k, Points):
  res = 0
  lst = np.where((Points[0, ] == k)&(Points[1,] < t) &(Points[1,] >= t-a) & (Points[3,]==1))[0]
  #print(lst)
  if len(lst) >0 :
      lpoint = Points[1,lst].astype(float)
      res = omega[k, i] * sum(np.exp(-alpha*(t- lpoint))) 
  return(res)





##Preselection
print("PRESELECTION")

Points=np.zeros((9,0))
idInPoints = -1

for i in AllId:
  tnext = tstart + rexp(1, majorante[i])
  while tnext < tend:
      L = np.floor(tnext/delta)
      
      ##the weights lamb_i in Kalikow decomp.
      Wts = lamb[i,]
      
      ##the Alli of the neighborhood. Note that in case of refractory period, we always pick at least neighborhood i 
      indV = IntSample(AllId, Wts, 1)
      
      if indV != i :
        maj_neigh = 4*domi/(abs(indV-i)**2 * (1- np.exp(-aldel)))
      else: maj_neigh = mu/lamb[i,i] + Co*domi /(lamb[i,i]*(1- np.exp(-aldel)))
      
      u1 = runif(1, 0, 1)
      if u1 <= maj_neigh/ majorante[i] :
        idInPoints = idInPoints +1
        #newpoints = [i, tnext, indV, -1 , -1, -1,  idInPoints, maj_neigh, u1].astype(float)
        Points = np.c_[Points, [i, tnext, indV, -1 , -1, -1,  idInPoints, maj_neigh, u1]]    ##first -1 means the mark has not decided yet (Forward)                                                              ##where the second is designed for Backward steps
                                                #X , #Y, #Z
      ##we need iInPoints for the Backward, Majorant_vj for Forward
      
      tnext = tnext + rexp(1, majorante[i])  

## END PRESELECTION##########################


#L = []
#count = 0

###BACKWARD##############

print("BACKWARD_FORWARD")

AcceptedPoints = [[-delta] for i in range(NbId)]  ##At the beginning we set T^k_0 =0 for all neuron k
##Co the Cai tien bang dung mot array la du

for i in AllId:
    indP = np.where((Points[0,] ==i) & (Points[3,] ==  -1) )[0]
    #lp = len(indP)
    for j in indP:
        ClanA = np.zeros((9, 0))
        ClanA = np.c_[ClanA, Points[:,j]]#list all the points needed to decide Points[,indP[j]]
        ##The use of new list ClanA is neccessary because we don't create a list of points like Patricia'code, 
        ##but we want to LIST all the points in the existing points (Points) in the Presection                               
        Points[5,j] = 1 ## mark Z = 1, mark that this point is already in ClanA
        #indPx = np.where((ClanA[4,] ==  -1))[0]  ##Find all the points which we did not do Backward
        #print(indPx)
        lstClanA = [0]  #Di nhien diem ta vua them la chua khoi tao Clan va id cua no trong ClanA la 0
        lenClanA = 1
        while lenClanA > 0 :
            for cx in lstClanA:
                IdNeigh = ClanA[2, cx]
                IdPoints = ClanA[0, cx]
                IdPointsInNeigh1 = np.where((Points[0,] == IdNeigh) & (Points[1,] < ClanA[1, cx]) &(Points[1,] >= ClanA[1, cx] - a)& (Points[5,] == -1))[0]
                IdPointsInNeigh2 = np.where((Points[0,] == IdPoints) & (Points[1,] < ClanA[1, cx]) & (Points[1,] > ClanA[1,cx] - delta) & (Points[5,] == -1))[0]
                #print("REFRACTORY###################", IdPointsInNeigh2)
                #Due to refractory period, the neighborhood of neuron i always contain the index i itself.
                
                IdPointsInNeigh = Union(IdPointsInNeigh1, IdPointsInNeigh2)
                l = Points[:,IdPointsInNeigh]
                ##list all the id of points that belongs to neighborhood and those never touched before (Points[5,] == -1)  or Z == -1.
                
                ClanA = np.c_[ClanA, l]
                Points[4, ClanA[6, cx]] = 1   ##mark Y = 1   ## ClanA[6,] returns a true id in Points
                ClanA[4, cx] = 1  ##mark Y = 1
                ## Mark that this point is already expanded (do Backward), and hence no longer need to reconsider
                Points[5,IdPointsInNeigh] = 1
                ## mark that all these points already in ClanA (Z = 1)
                
                #print(ClanA)
            lstClanA = np.where((ClanA[4,] ==  -1))[0] 
            lenClanA = len(lstClanA)
            print("-----------------------------------------------------")
        print("I'am the point", Points[:,j])
        print("My Clan is", ClanA)
        ##end of Backward
        ##return ClanA that contains all the points needed to decie Points[,indP[j]]
      
  ##In forward, we will decide the state of all the points in ClanA in increasing time and update its state in Points
  ##Forward
  ##Note that, most of the time in Forward, we only need to sort a few points, so the algorithm goes fast.

        ind_thin = np.where(ClanA[3,] == -1)[0]
        #tri=sort(ClanA[1,ind_thin],index.return=TRUE)
        ind_sorted = ClanA[1,ind_thin].argsort()  
        #print("Sorted Index")
        #print(ind_sorted)
        #good= ind_thin[tri$ix] #the good order to take the points into ClanA
        #lp_sorted = len(ind_sorted)
        
        for ind in ind_sorted:
              #ind = good[ind]   ##index before sorting
              iV = ClanA[0,ind]
              jV = ClanA[2,ind]
              tV = ClanA[1,ind]
              #print("Begin FORWARD")
              phi_neigh =0
              #print("LENGTH OF ACCEPTEDPOINTS of ID",iV, "is---", len(AcceptedPoints[iV]))
           
              lastspike = AcceptedPoints[iV][-1]
              #if lastspike != -delta:
                  #print("Here is the last spike of index", iV, "that is", lastspike)
              if (tV - lastspike > delta) :
                  #print("The condition PASSED")
                  if jV == iV :
                        phi_neigh = (mu+ interact(tV, iV, jV, Points)) / lamb[iV,iV]   ##Note that we don't have empty neighborhood
                  else:
                        phi_neigh =  interact(tV, iV, jV, Points) / lamb[iV, jV]
              #L.append(phi_neigh)
              #print("phi of neighborhood", phi_neigh)
              if ClanA[8, ind] <= phi_neigh/majorante[iV] :
              #if ClanA[8, ind] <= phi_neigh/ClanA[7, ind] :
                ##update mark
                Points[3, ClanA[6,ind]] = 1 ##Note that we need to use ClanA[6,] to return a true position in Points
                x = float(Points[1, ClanA[6,ind]])
                AcceptedPoints[iV].append(x)
                #print("I accept this point, now the list is")
                #print("List of Accepted Points of index", iV)
                #print(AcceptedPoints[iV])
                #sys.exit
              else:
                #print("I reject this point")
                Points[3, ClanA[6,ind]] = 0
        print("###################################################")
AccPoints = Points[:2, Points[3,] == 1]      
print(AccPoints)

np.savetxt('/Users/phicuong/Downloads/Kalikow/test_OgataKalikowClan.csv', AccPoints, delimiter = ',')

