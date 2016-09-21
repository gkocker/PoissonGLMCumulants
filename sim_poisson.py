# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:04:49 2015

@author: gabeo

simulate a network of Poisson neurons with alpha-function synapses

"""
import os
import numpy as np
from phi import phi
from phi import phi_prime
from theory import rates_ss
import params
from generate_adj import generate_adj as gen_adj
from correlation_functions import bin_pop_spiketrain, auto_covariance_pop

def sim_poisson(W, tstop, trans, dt):

    '''
    :param W: weight matrix
    :param tstop: total simulation time (including initial transient)
    :param trans: initial transient (don't record spikes until trans milliseconds)
    :param dt: Euler step
    :return:
    '''
    
    # unpackage parameters
    par = params.params()
    N = par.N
    tau = par.tau
    b = par.b
    gain = par.gain

    # sim variables
    Nt = int(tstop/dt)
    t = 0
    numspikes = 0
    
    maxspikes = 500*N*tstop/1000  # 500 Hz / neuron
    spktimes = np.zeros((maxspikes, 2)) # store spike times and neuron labels
    g_vec = np.zeros((Nt, N))

    # alpha function synaptic variables
    s = np.zeros((N,))
    s0 = np.zeros((N,))
    s_dummy = np.zeros((N,))
    
    a = 1. / tau
    a2 = a**2
    
    for i in range(0,Nt,1):
        
        t += dt
        
        # update each neuron's output and plasticity traces
        s_dummy += dt*(-2*a*s_dummy - a2*s)    
        s = s0 + dt*s_dummy
        
        # compute each neuron's input
        g = np.dot(W, s) + b
        g_vec[i] = g

        # decide if each neuron spikes, update synaptic output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain)

        try:
            spiket = np.random.poisson(r*dt, size=(N,))
        except:
            break

        s_dummy += spiket*a2  # a2 for alpha function normalized to have unit integral
    
        ### store spike times and counts
        if t > trans:
            for j in range(N):
                if spiket[j] >= 1 and numspikes < maxspikes:
                    spktimes[numspikes, 0] = t
                    spktimes[numspikes, 1] = j
                    numspikes += 1
        
        s0 = s
    
    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]

    return spktimes, g_vec


if __name__ == '__main__':

    par = params.params()
    
    Ne = par.Ne
    Ni = par.Ni
    N = par.N
    pEE = par.pEE
    pEI = par.pEI
    pIE = par.pIE
    pII = par.pII
    weightEE = par.weightEE
    weightEI = par.weightEI
    weightIE = par.weightIE
    weightII = par.weightII
    tau = par.tau
    b = par.b
    gain = par.gain

    trans = 5. * tau  # simulation transient
    tstop = 4000. * tau + trans  # simulation time
    dt = .2 * tau  # Euler step

    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII) # generate adjacency matrix

    W = W0.copy() # generate weight matrix
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE*W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE*W0[Ne:, 0:Ne]
        
    if Ni > 0:
        W[0:Ne,Ne:] = weightEI*W0[0:Ne, Ne:]
        W[Ne:,Ne:] = weightII*W0[Ne:, Ne:]

    spktimes, g_vec = sim_poisson(W, tstop, trans, dt)

    # compute some statistics
    ind_include = range(Ne)  # indices of E neurons
    spk_Epop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, ind_include)
    dt_ccg = 1.  # ms
    lags = np.arange(-10.*tau, 10.*tau, dt_ccg)
    pop_2point = auto_covariance_pop(spktimes, ind_include, numspikes, dt, lags, tau, tstop, trans)



