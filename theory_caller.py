# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:38:45 2015

@author: gabeo
"""
import numpy as np
import params; reload(params)
from generate_adj import generate_adj as gen_adj
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
W = W0
if Ne > 0:
    W[0:,0:Ne] = weightE*W0[0:,0:Ne]
    W[0,1] = weightE*.8
if Ni > 0:
    W[0:,Ne:N] = weightI*W0[0:,Ne:N]
    
Tmax = 10*tau

print 'computing second cumulants (predicted)'
from theory import two_point_function
t_th, C2 = two_point_function(W)

from theory import two_point_function_1loop
t_th_1loop, C2_1loop = two_point_function_1loop(W)

Nt = t_th.shape[0]
#
print 'running sims'
import sim_poisson
tstop = 2000000*tau
trans = 5*tau
dt = .02*tau
Tmax = 100
spktimes = sim_poisson.sim_poisson(W, tstop, trans, dt)

from correlation_functions import cross_covariance_spk
print 'computing second cumulants (empirical)'

lags = np.arange(-Tmax,Tmax+1,1)
Nlags = lags.size
xcov2 = np.zeros((N,N,Nlags))
for i in range(N):
    for j in range(N):
        xcov2[i,j,:] = cross_covariance_spk(spktimes,spktimes.shape[0],i,j,dt,lags,tau,tstop,trans)    
##
##
plt.figure();
plt.plot(lags,xcov2[0,1,:],'o')
plt.plot(t_th,C2[0,1,:],'k',t_th,C2[0,1,:]+C2_1loop[0,1,:],'r',linewidth=2)
plt.xlim((-Tmax,Tmax))

plt.figure();
plt.plot(lags,xcov2[1,12,:],'o')
plt.plot(t_th,C2[1,12,:],'k',t_th,C2[1,12,:]+C2_1loop[1,12,:],'r',linewidth=2)
plt.xlim((-Tmax,Tmax))

plt.figure();
plt.plot(lags,xcov2[11,12,:],'o')
plt.plot(t_th,C2[11,12,:],'k',t_th,C2[11,12,:]+C2_1loop[11,12,:],'r',linewidth=2)
plt.xlim((-Tmax,Tmax))

#print 'computing third cumulants (predicted)'
#from theory import three_point_function
#t_th1, t_th2, C3 = three_point_function(W)
#
#Nt = t_th1.shape[0]
#plt.figure();
#plt.plot(t_th1, C3[0,1,0,:,Nt/2])
#
#fig = plt.figure()
#ax = plt3.Axes3D(fig)
#X, Y = np.meshgrid(t_th1, t_th2)
#ax.plot_surface(X,Y,C3[0,1,0,:,:])
#ax.set_ylabel('Time lag, s1 (ms)')
#ax.set_xlabel('Time lag, s2 (ms)')
#ax.set_zlabel('Third cross-cumulant, cells 0,1,0 (sp/ms)'+r'$^3$')
#ax.set_xlim((-100,100))
#ax.set_ylim((-100,100))

#plt.figure();
#plt.plot(t_th1, C3[1,0,1,Nt/2,:])