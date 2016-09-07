# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:35:03 2015

@author: gabeo

Euler scheme for theoretical prediction of weight dynamics
Here, implement just the second-order contribution- the pair-based depression, without the triplet-based potentiation
"""

import numpy as np
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

A3plus = 0
#A2minus = 0 # keep pairwise term

### generate adjacency matrix
#W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)
W0 = np.zeros((N,N))
W0[0,1] = 1
W0[1,0] = 1

### generate weight matrix
W = np.dot(W0, np.eye(N))
if Ne > 0:
    W[0:,0:Ne] = weightE*W0[0:,0:Ne]
    W[1,0] = weightE/2
if Ni > 0:
    W[0:,Ne:N] = weightI*W0[0:,Ne:N]

tstop = 100000
Nt = 20
dt = tstop/Nt

Wt = np.zeros((N,N,Nt))
Wt[:,:,0] = W
print A2minus

for i in range(1,Nt):
    
    r0 = theory.rates_ss(W)
    t_th, C2 = theory.two_point_function(W, 100, 1)
    C2 = C2.real
    ind0 = np.where(abs(t_th)==np.amin(abs(t_th)))[0][0]
        
    stdp_window = np.zeros(t_th.shape)
    stdp_window[0:ind0] = -A2minus*np.exp(-np.abs(t_th[0:ind0])/tauminus)
    dt_th = t_th[1]-t_th[0]
    
    dWdt = np.zeros((N,N))    
    for ii in range(0,N):
        for jj in range(0,N):
            dWdt[ii,jj] = W0[ii,jj]*np.sum((r0[ii]*r0[jj]+C2[ii,jj,:])*stdp_window)*dt_th    
#            if ii == jj:
#                dWdt[ii,jj] = 0
            if W[ii,jj]+dWdt[ii,jj] < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
    
    Wt[:,:,i] = Wt[:,:,i-1] + dWdt*dt
    W = W+dWdt*dt
    


tplot = np.arange(0,tstop,dt)
plt.figure(); plt.plot(tplot,Wt[1,0,:],'k',linewidth=2); plt.ylim((0,WmaxE)); plt.xlim((0,tstop))