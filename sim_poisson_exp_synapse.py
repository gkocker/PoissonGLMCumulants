# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:04:49 2015

@author: gabeo

simulate a network of Poisson neurons
"""
import numpy as np
from phi import phi
import params; reload(params)
from generate_adj import generate_adj as gen_adj
import matplotlib.pyplot as plt

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

# generate adjacency matrix
W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)

# generate weight matrix
W = np.dot(W0, np.eye(N))
if Ne > 0:
    W[0:,0:Ne] = weightE*W0[0:,0:Ne]
    W[0,1] = weightE*2
if Ni > 0:
    W[0:,Ne:N] = weightI*W0[0:,Ne:N]

#W0 = np.load('W0.npy')
#W = np.load('W.npy')

dt = .02*tau
trans = 5.*tau
tstop = 100000.*tau + trans
Nt = int(tstop/dt)
Ntrans = int(trans/dt)
t = 0
numspikes = 0

count = np.zeros((N,)) # spike count of each neuron in the interval (0,tstop)
maxspikes = 100*N*tstop/1000 # 100 Hz / neuron
spktimes = np.zeros((maxspikes,2)) # spike times and neuron labels

Tmax = 5.*tau

s = np.zeros((N,)) # synaptic output of each neuron
s0 = np.zeros((N,))

for i in range(0,Nt,1):
    
    t +=dt    
    
    # update each neuron's output and plasticity traces
    s = s0 + dt*(-1./tau*s0)
    
    # compute each neuron's input
    g = np.dot(W,s) + b
    
    # decide if each neuron spikes, update output of spiking neurons
    # each neurons's rate is phi(g)
    r = phi(g,gain)
    
    spiket = np.random.rand(N,)
    spiket[spiket<r*dt] = 1
    spiket[spiket!=1] = 0
    
    s += spiket

    ### store spike times and counts
    if t > trans:
        count += spiket
        if numspikes < maxspikes:
            for j in range(0,N):
                if spiket[j] == 1:
                    spktimes[numspikes,0] = t
                    spktimes[numspikes,1] = j
                    numspikes += 1
    
    s0 = s

# truncate spike time array
spktimes = spktimes[0:numspikes,:]


### exmpirical convariance
#from correlation_functions import cross_covariance_spk
#print 'computing second cumulants (empirical)'

#lags = np.arange(-Tmax,Tmax+1,1)
#Nlags = lags.size
#xcov2 = np.zeros((N,N,Nlags))
#for i in range(N):
#    for j in range(N):
#        xcov2[i,j,:] = cross_covariance_spk(spktimes,numspikes,i,j,dt,lags,tau,tstop,trans)
                
                
### plot raster
#from raster import raster
#tplot = 5000
#spktimes_plot = spktimes[np.where(spktimes[:,0]<tplot)[0],:]
#numspikes_plot = np.shape(spktimes_plot)[0]
#raster(spktimes_plot,numspikes_plot)

### total triplet autocorrelation
#print 'computing total triplet autocovariances (empirical)'
#from correlation_functions import triplet_covariance_tot
#from correlation_functions import cross_covariance_tot
#lags = np.arange(-Tmax,Tmax+1,dt_ccg)
#autocov2_tot = np.zeros((N,))
#autocov3_tot = np.zeros((N,))
#for n in range(N):
#    autocov2_tot[n] = cross_covariance_tot(spktimes,numspikes,n,n,dt,lags,tau,tstop,trans)
#    autocov3_tot[n] = triplet_covariance_tot(spktimes,numspikes,n,n,n,dt,lags,tau,tstop,trans)


### third cumulants
print 'computing third cumulants (empirical)'
from correlation_functions import triplet_covariance_spk
lags = np.arange(-Tmax,Tmax+1,1)
Nlags = lags.size
xcov3 = np.zeros((N,N,N,Nlags,Nlags))
for i in range(N):
    for j in range(N):
        for k in range(N):
                xcov3[i,j,k,:,:] = triplet_covariance_spk(spktimes,numspikes,i,j,k,dt,lags,tau,tstop,trans)
                
plt.figure(); 
plt.plot(lags,xcov3[0,1,0,:,50],'o')

plt.figure(); 
plt.plot(lags,xcov3[1,0,1,:,50],'o')

plt.figure();
plt.plot(lags,xcov3[0,1,1,50,:],'o')