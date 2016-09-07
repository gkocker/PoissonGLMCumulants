# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:01:48 2016

@author: gabeo

Figure: distribution of population spike counts affects rate of a readout neuron
panels: distribution of 
"""

from generate_adj import generate_adj as gen_adj
import params; reload(params)
import sim_poisson
import matplotlib.pyplot as plt
import numpy as np
from correlation_functions import bin_spiketrain
from correlation_functions import bin_pop_spiketrain
from correlation_functions import count_dist
from phi import phi
from phi import phi_prime
from theory import rates_ss

# unpackage parameters
par = params.params()
Ne = par.Ne
Ni = par.Ni
N = par.N
pEE = par.pEE
pEI = par.pEI
pIE = par.pIE
pII = par.pII
tau = par.tau
b = par.b
gain = par.gain
weightEE = par.weightEE
weightEI = par.weightEI
weightIE = par.weightIE
weightII = par.weightII


tstop = 50. * tau
dt = .02 * tau     
trans = 5. * tau
 
fig, ax = plt.subplots(2,2,figsize=(20,10))     

window = 100
 
# generate adjacency matrix
W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)
# make first neuron a readout
W0[0,1:] = 1
W0[:,0] = 0

W = np.dot(W0, np.eye(N))
if Ne > 0:
    W[0:Ne,0:Ne] = weightEE*W0[0:Ne,0:Ne]
    W[Ne:,0:Ne] = weightIE*W0[Ne:,0:Ne]
    
if Ni > 0:
    W[0:Ne,Ne:] = weightEI*W0[0:Ne,Ne:]
    W[Ne:,Ne:] = weightII*W0[Ne:,Ne:]

r_th = rates_ss(W)    
g = np.dot(W,r_th) + b
stab_matrix = np.dot(np.diag(phi_prime(g, gain)), W)
spec_rad_1 = max(abs(np.linalg.eigvals(stab_matrix)))

spktimes = sim_poisson.sim_poisson(W, spec_rad_desired, tstop, trans, dt)

raster(spktimes, spktimes.shape[0])

ind_include = range(1,N)
spk = bin_pop_spiketrain(spktimes,dt,1,tstop,trans,ind_include)
count = count_dist(spk, 1, tstop, window)
#
#plt.figure(); plt.hist(count,bins=40)
#plt.title('Presynaptic spike count distribution')
#

spk_post = bin_spiketrain(spktimes,0,dt,1,tstop,trans)
#
#print sum(spk_post)/(tstop-trans)*1000
count_post = count_dist(spk_post, 1 , tstop, window)
#plt.figure(); plt.hist(count_post,bins=10)
#plt.title('Postsynaptic spike count distribution')    

ax[0,0].hist(count, bins=20, normed=1)
ax[1,0].hist(count_post, bins=10, normed=1)
ax[1,0].set_xlabel('Spike count')    
ax[0,0].set_ylabel('Presynaptic population')
ax[1,0].set_ylabel('Readout neuron')
ax[0,0].set_title('Mean pre count: '+str(np.mean(count)))
ax[1,0].set_title('Mean post count: '+str(np.mean(count_post)))


W *= 2
r_th = rates_ss(W)    
g = np.dot(W,r_th) + b
stab_matrix = np.dot(np.diag(phi_prime(g, gain)), W)
spec_rad_2 = max(abs(np.linalg.eigvals(stab_matrix)))

spktimes = sim_poisson.sim_poisson(W, spec_rad_desired, tstop, trans, dt)

ind_include = range(1,N)
spk = bin_pop_spiketrain(spktimes,dt,1,tstop,trans,ind_include)
count = count_dist(spk, 1, tstop, window)
#
#plt.figure(); plt.hist(count,bins=40)
#plt.title('Presynaptic spike count distribution')
#

spk_post = bin_spiketrain(spktimes,0,dt,1,tstop,trans)
#
#print sum(spk_post)/(tstop-trans)*1000
count_post = count_dist(spk_post, 1 , tstop, window)
#plt.figure(); plt.hist(count_post,bins=10)
#plt.title('Postsynaptic spike count distribution')    

ax[0,1].hist(count, bins=20, normed=1)
ax[1,1].hist(count_post, bins=10, normed=1)
ax[1,1].set_xlabel('Spike count')    
ax[0,1].set_title('Mean pre count: '+str(np.mean(count)))
ax[1,1].set_title('Mean post count: '+str(np.mean(count_post)))

ax[0,0].set_xlim((0,1.1*max(count)))
ax[0,1].set_xlim((0,1.1*max(count)))
ax[1,0].set_xlim((0,1.1*max(count_post)))
ax[1,1].set_xlim((0,1.1*max(count_post)))


