# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:10:17 2015

@author: gabeo

plot synaptic filter

"""

import numpy as np
import matplotlib.pyplot as plt
import params; reload(params)

par = params.params()
tau = par.tau 

tstop = 100
dt = .05
Nt = int(tstop/dt)

s = np.zeros((Nt,))
s[0] = 1

r = 10;

for i in range(1,Nt):
    s[i] = s[i-1]+dt*(-s[i-1]/tau)
    
    spike = np.random.rand(1)
    if spike < r*dt:
        spike = 1
    else:
        spike = 0
        
    s[i] += spike
    
    
fig_syn = plt.figure()
tplot = np.arange(0,tstop,dt)
plt.plot(tplot,s,'k')