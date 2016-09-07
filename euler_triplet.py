# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:35:03 2015

@author: gabeo

Euler scheme for theoretical prediction of weight dynamics
Here, implement just the second-order contribution- the pair-based depression, without the triplet-based potentiation
"""

import numpy as np
from phi import phi
import params; reload(params)
from generate_adj import generate_adj as gen_adj
import matplotlib.pyplot as plt
import math
import theory

# unpackage parameters
par = params.params()

Ne = par.Ne
Ni = par.Ni
N = par.N
pEE = par.pEE
pEI = par.pEI
pIE = par.pIE
pII = par.pII
weightE = par.weightEE
weightI = par.weightII
tau = par.tau
b = par.b
gain = par.gain
A3plus = par.A3plus
A2minus = par.A2minus
tauplus = par.tauplus
tauminus = par.tauminus
taux = par.taux
tauy = par.tauy
WmaxE = par.WmaxE
eta = par.eta

A3plus = eta*A3plus
A2minus = eta*A2minus

#A3plus = 0
#A2minus = 0 # keep pairwise term

### generate adjacency matrix
#W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)
#W0 = np.zeros((N,N))
#W0[0,1] = 1
#W0[1,0] = 1
#if N==3:
#    W0[0,2] = 1
#    W0[1,2] = 1
##
#### generate weight matrix
#W = np.dot(W0, np.eye(N))
#if Ne > 0:
#    W[0:,0:Ne] = weightE*W0[0:,0:Ne]
##    W[1,0] = weightE/3
#    if Ne==3:
#        W[1,2] = weightE/2
#if Ni > 0:
#    W[0:,Ne:N] = weightI*W0[0:,Ne:N]

W0 = np.load('W0.npy')
W = np.load('W.npy')

tstop = 440000
Nt = 1
dt = tstop/Nt

Wt = np.zeros((N,N,Nt))
Wt[:,:,0] = W
print A3plus
#
#ind = np.array([[0],[1]])
#C3f, w = theory.three_point_function_fourier(W, 100, 1)
#t_th1, t_th2, C3 = theory.three_point_function(W, 100, 1)
#

for i in range(1,Nt):
    
    print i    
    
    r0 = theory.rates_ss(W)
    t_th, C2 = theory.two_point_function(W, 100, 1)
    C2 = C2.real
    ind_cut = t_th.size/4
    C2 = C2[:,:,ind_cut:-ind_cut]
    t_th = t_th[ind_cut:-ind_cut]
    ind0 = np.where(abs(t_th)==np.amin(abs(t_th)))[0][0]
        
    L = np.zeros(t_th.shape)
    L[0:ind0] = -A2minus*np.exp(-np.abs(t_th[0:ind0])/tauminus)
       
    dt_th = t_th[1]-t_th[0]
    
    t_th1, t_th2, C3 = theory.three_point_function(W, 100, 1)
    
    C3 = C3.real
    ind_cut = t_th1.size/4
    C3 = C3[:,:,:,ind_cut:-ind_cut,ind_cut:-ind_cut]
    t_th1 = t_th1[ind_cut:-ind_cut]    
    t_th2 = t_th2[ind_cut:-ind_cut]    
    
    dt_th1 = t_th1[1]-t_th1[0]
    dt_th2 = t_th2[1]-t_th2[0]
    ind0 = np.where(abs(t_th1)==np.amin(abs(t_th1)))[0][0]    
    
    Q = np.zeros((t_th1.size, t_th2.size))     
    Q[ind0:,ind0:] = A3plus*np.outer( np.exp(-np.abs(t_th1[ind0:])/tauplus),np.exp(-np.abs(t_th2[ind0:])/tauy) )
    Q[ind0,:] = 0 # tpost2 = tpost 1 means a spike pair, included in the pair-based rule
    
    Nt_th  = t_th.size
#    indt = range(int(np.floor(Nt_th/4)), Nt_th-int(np.floor(Nt_th/4)))
    indt = range(0,Nt_th,2)
    
    C2_lagdiff = np.zeros((N,N,len(indt),len(indt)))
    for s1 in range(0,len(indt)):
        for s2 in range(0,len(indt)):
#            C2_lagdiff[:,:,s1,s2] = C2[:,:,indt[s1]-indt[s2]]
            C2_lagdiff[:,:,s1,s2] = C2[:,:,s1-s2]
    
    dWdt = np.zeros((N,N))    
    for ii in range(0,Ne):
        for jj in range(0,Ne):
            dWdt[ii,jj] += W0[ii,jj]*np.sum((r0[ii]*r0[jj]+C2[ii,jj,:])*L)*dt_th    # pair-based depression
            dWdt[ii,jj] += W0[ii,jj]*np.sum( np.sum((r0[ii]*r0[ii]*r0[jj] + r0[ii]*C2[ii,jj,indt] + r0[ii]*C2_lagdiff[ii,jj,:,:] + r0[jj]*C2[ii,ii,indt] + C3[ii,ii,jj,:,:])*Q, 0)*dt_th1 )*dt_th2  # triplet-based potentiation
#            dWdt[ii,jj] += W0[ii,jj]*np.sum( np.sum((r0[ii]*r0[ii]*r0[jj] + r0[ii]*C2[ii,jj,indt] + r0[ii]*C2_lagdiff[ii,jj,:,:] + r0[jj]*C2[ii,ii,indt])*Q, 0)*dt_th1 )*dt_th2  # triplet-based potentiation

   #            if ii == jj:
    #                dWdt[ii,jj] = 0
            if W[ii,jj]+dWdt[ii,jj] < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
            if W[ii,jj]+dWdt[ii,jj] > WmaxE:  
                dWdt[ii,jj] = (WmaxE - W[ii,jj])/dt
    
    Wt[:,:,i] = Wt[:,:,i-1] + dWdt*dt
    W = W + dWdt*dt
    
    print dWdt

#

#tplot = np.arange(0,tstop,dt)
#plt.figure(); 
#plt.plot(tplot,Wt[0,1,:],'m',tplot,Wt[1,0,:],'m',linewidth=2)
#plt.plot(tplot,Wt[0,2,:],'m',tplot,Wt[1,2,:],'m',linewidth=2)
#plt.ylim([0,WmaxE]); plt.xlim((0,tstop))
#
#
C2f, w = theory.two_point_function_fourier(W,100,1)
t_th, C2 = theory.two_point_function(W,100,1)
#plt.figure(); plt.plot(w,C2f[0,1,:],'b',w,C2f[1,0,:],'g')
ind_1 = 0
ind_2 = Ne + np.where(W0[ind_1,Ne:]==1)[0][0]
plt.figure(); plt.plot(t_th,C2[ind_1,ind_2,:],'b',linewidth=2)
plt.xlim((-100,100))
#
#t_th1, t_th2, C3 = theory.three_point_function(W,100,1)
#plt.figure(); plt.plot(t_th1,C3[0,0,1,150,:])
#plt.figure(); plt.plot(t_th1,C3[0,0,1,500,:])

#C3f, w3 = theory.three_point_function_fourier(W,100,1)
#plt.figure(); plt.plot(w3,C3f[0,1,1,250,:])