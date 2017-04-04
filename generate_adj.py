# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:16:21 2015

@author: gabeo

generate non-symmetric Erdos-Renyi adjacency matrix with E/I block structure

dependency: numpy
"""
import numpy as np

def generate_adj(Ne,Ni,pEE,pEI,pIE,pII):

    N = Ne+Ni
    
    W0EE = np.random.rand(Ne,Ne)
    W0EI = np.random.rand(Ne,Ni)
    W0IE = np.random.rand(Ni,Ne)
    W0II = np.random.rand(Ni,Ni)
    
    maskEE = W0EE<pEE
    maskEI = W0EI<pEI
    maskIE = W0IE<pIE
    maskII = W0II<pII
    
    W0EE[maskEE] = 1
    W0EE[np.logical_not(maskEE)] = 0
    W0EI[maskEI] = 1
    W0EI[np.logical_not(maskEI)] = 0
    W0IE[maskIE] = 1
    W0IE[np.logical_not(maskIE)] = 0
    W0II[maskII] = 1
    W0II[np.logical_not(maskII)] = 0
    
    if Ni > 0:
        W0 = np.concatenate((W0EE,W0EI),1)
        W0= np.concatenate((W0,np.concatenate((W0IE,W0II),1)),0)
    else:
        W0 = W0EE
    
    np.fill_diagonal(W0,0) # disallow autapses    
    
    return W0

def generate_regular_adj(Ne,Ni,pEE,pEI,pIE,pII):

    N = Ne+Ni
    kEE = pEE*Ne
    kEI = pEI*Ni
    kIE = pIE*Ne
    kII = pII*Ni

    W0 = np.zeros((N, N))

    for n in range(Ne):
        other_Eneurons = np.concatenate((np.arange(n), np.arange(n+1, Ne)))
        ind_E = np.random.choice(other_Eneurons, size=kEE, replace=False)
        ind_I = np.random.choice(range(Ne, N), size=kEI, replace=False)
        W0[n, ind_E] = 1.
        W0[n, ind_I] = 1.

    for n in range(Ne, N):
        ind_E = np.random.choice(range(Ne), size=kIE, replace=False)
        other_Ineurons = np.concatenate((np.arange(Ne, n), np.arange(n+1, N)))
        ind_I = np.random.choice(other_Ineurons, size=kII, replace=False)

        W0[n, ind_E] = 1.
        W0[n, ind_I] = 1.

    return W0