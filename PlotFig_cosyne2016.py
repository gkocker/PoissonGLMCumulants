# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:10:22 2015

@author: gocker

CoSyne 2016 abstract figures
here, just generate each panel - put them together in Illustrator
"""

""" panel 1: dWdt vs W """

import numpy as np
from phi import phi
import params; reload(params)
from generate_adj import generate_adj as gen_adj
import matplotlib.pyplot as plt
import theory

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
#

Nplot = 50
Wvec = np.linspace(WmaxE*.01,WmaxE*.95,Nplot)

r_vec = np.zeros((Nplot,))
dW_1 = np.zeros((Nplot,)) # rate contribution
dW_2 = np.zeros((Nplot,)) # rate and two-point contributions
dW_3 = np.zeros((Nplot,)) # rate and two-point and three-point contributions
#
#### weight change in a symmetric, reciprocally coupled pair of neuronss
for i in range(0,Nplot):
    
    W = Wvec[i]*W0    
    
    r0 = theory.rates_ss(W)
    t_th, C2 = theory.two_point_function(W, 100, 1)
    C2 = C2.real
    ind0 = np.where(abs(t_th)==np.amin(abs(t_th)))[0][0]
        
    L = np.zeros(t_th.shape)
    L[0:ind0] = -A2minus*np.exp(-np.abs(t_th[0:ind0])/tauminus)
    
    dt_th = t_th[1]-t_th[0]
    
    t_th1, t_th2, C3 = theory.three_point_function(W, 100, 1)
    C3 = C3.real
    dt_th1 = t_th1[1]-t_th1[0]
    dt_th2 = t_th2[1]-t_th2[0]
    ind0 = np.where(abs(t_th1)==np.amin(abs(t_th1)))[0][0]    
    
    Q = np.zeros((t_th1.size, t_th2.size))     
    Q[ind0:,ind0:] = A3plus*np.outer( np.exp(-np.abs(t_th1[ind0:])/tauplus),np.exp(-np.abs(t_th2[ind0:])/tauy) )
    Q[ind0,:] = 0 # tpost2 = tpost 1 means a spike pair, included in the pair-based rule

    Nt_th  = t_th.size
    indt = range(0,Nt_th,2)
    
    C2_lagdiff = np.zeros((N,N,len(indt),len(indt)))
    for s1 in range(0,len(indt)):
        for s2 in range(0,len(indt)):
            C2_lagdiff[:,:,s1,s2] = C2[:,:,indt[s1]-indt[s2]]
    
    r_vec[i] = r0[0]
#    dW_1[i] += W0[ii,jj]*np.sum((r0[ii]*r0[jj]+C2[ii,jj,:])*L)*dt_th  
    
    
    dW_1[i] = W0[0,1]*np.sum((r0[0]*r0[1])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1])*Q, 0)*dt_th1 )*dt_th2
    dW_2[i] = W0[0,1]*np.sum((r0[0]*r0[1]+C2[0,1,:])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1] + r0[0]*C2[0,1,indt] + r0[0]*C2_lagdiff[0,1,:,:] + r0[1]*C2[0,0,indt])*Q, 0)*dt_th1 )*dt_th2
    dW_3[i] = W0[0,1]*np.sum((r0[0]*r0[1]+C2[0,1,:])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1] + r0[0]*C2[0,1,indt] + r0[0]*C2_lagdiff[0,1,:,:] + r0[1]*C2[0,0,indt] + C3[0,0,1,:,:])*Q, 0)*dt_th1 )*dt_th2
    

dW_symm = plt.figure(); 
plt.plot(Wvec,dW_1,'b',Wvec,dW_2,'k',Wvec,dW_3,'r',linewidth=2)
plt.plot(Wvec,np.zeros(Nplot,),'k',linewidth=.5)
plt.xlabel('Synaptic weight, W'); plt.ylabel('Plasticity dynamics, dW')

### quiver plot
Nquiv = 8
Wvec_quiv = np.linspace(WmaxE*.01,WmaxE*.9,Nquiv)
r_quiv = np.zeros((Nquiv,Nquiv))

dW01quiv_1 = np.zeros((Nquiv,Nquiv))
dW01quiv_2 = np.zeros((Nquiv,Nquiv))
dW01quiv_3 = np.zeros((Nquiv,Nquiv))

dW10quiv_1 = np.zeros((Nquiv,Nquiv))
dW10quiv_2 = np.zeros((Nquiv,Nquiv))
dW10quiv_3 = np.zeros((Nquiv,Nquiv))


for i in range(0,Nquiv):
    W[0,1] = Wvec_quiv[i]
    for j in range(0,Nquiv):
        W[1,0] = Wvec_quiv[j]
        
        
        r0 = theory.rates_ss(W)
        t_th, C2 = theory.two_point_function(W, 100, 1)
        C2 = C2.real
        ind0 = np.where(abs(t_th)==np.amin(abs(t_th)))[0][0]
            
        L = np.zeros(t_th.shape)
        L[0:ind0] = -A2minus*np.exp(-np.abs(t_th[0:ind0])/tauminus)
        
        dt_th = t_th[1]-t_th[0]
        
        t_th1, t_th2, C3 = theory.three_point_function(W, 100, 1)
        C3 = C3.real
        dt_th1 = t_th1[1]-t_th1[0]
        dt_th2 = t_th2[1]-t_th2[0]
        ind0 = np.where(abs(t_th1)==np.amin(abs(t_th1)))[0][0]    
        
        Q = np.zeros((t_th1.size, t_th2.size))     
        Q[ind0:,ind0:] = A3plus*np.outer( np.exp(-np.abs(t_th1[ind0:])/tauplus),np.exp(-np.abs(t_th2[ind0:])/tauy) )
        Q[ind0,:] = 0 # tpost2 = tpost 1 means a spike pair, included in the pair-based rule

        Nt_th  = t_th.size
        indt = range(0,Nt_th,2)
        
        C2_lagdiff = np.zeros((N,N,len(indt),len(indt)))
        for s1 in range(0,len(indt)):
            for s2 in range(0,len(indt)):
                C2_lagdiff[:,:,s1,s2] = C2[:,:,indt[s1]-indt[s2]]
        
        dW01quiv_1[i,j] = W0[0,1]*np.sum((r0[0]*r0[1])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1])*Q, 0)*dt_th1 )*dt_th2
        dW01quiv_2[i,j] = W0[0,1]*np.sum((r0[0]*r0[1]+C2[0,1,:])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1] + r0[0]*C2[0,1,indt] + r0[0]*C2_lagdiff[0,1,:,:] + r0[1]*C2[0,0,indt])*Q, 0)*dt_th1 )*dt_th2
        dW01quiv_3[i,j] = W0[0,1]*np.sum((r0[0]*r0[1]+C2[0,1,:])*L)*dt_th + W0[0,1]*np.sum( np.sum((r0[0]*r0[0]*r0[1] + r0[0]*C2[0,1,indt] + r0[0]*C2_lagdiff[0,1,:,:] + r0[1]*C2[0,0,indt] + C3[0,0,1,:,:])*Q, 0)*dt_th1 )*dt_th2
    
        dW10quiv_1[i,j] = W0[1,0]*np.sum((r0[1]*r0[0])*L)*dt_th + W0[1,0]*np.sum( np.sum((r0[1]*r0[1]*r0[0])*Q, 0)*dt_th1 )*dt_th2
        dW10quiv_2[i,j] = W0[1,0]*np.sum((r0[1]*r0[0]+C2[1,0,:])*L)*dt_th + W0[1,0]*np.sum( np.sum((r0[1]*r0[1]*r0[0] + r0[1]*C2[1,0,indt] + r0[1]*C2_lagdiff[1,0,:,:] + r0[0]*C2[1,1,indt])*Q, 0)*dt_th1 )*dt_th2
        dW10quiv_3[i,j] = W0[1,0]*np.sum((r0[1]*r0[0]+C2[1,0,:])*L)*dt_th + W0[1,0]*np.sum( np.sum((r0[1]*r0[1]*r0[0] + r0[1]*C2[1,0,indt] + r0[1]*C2_lagdiff[1,0,:,:] + r0[0]*C2[1,1,indt] + C3[1,1,0,:,:])*Q, 0)*dt_th1 )*dt_th2
    
        r_quiv[i,j] = np.sqrt(r0[0]*r0[1])

dW1_quiv = plt.figure(); plt.quiver(Wvec_quiv,Wvec_quiv,dW01quiv_1,dW10quiv_1,angles='xy',color='b');
dW2_quiv = plt.figure(); plt.quiver(Wvec_quiv,Wvec_quiv,dW01quiv_2,dW10quiv_2,angles='xy',color='k');
dW3_quiv = plt.figure(); plt.quiver(Wvec_quiv,Wvec_quiv,dW01quiv_3,dW10quiv_3,angles='xy',color='r');