# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:32:39 2016

@author: gabeo
generate a network use the degree distribution method (Zhao et al., Front Comp Neuro 2011, Hu et al J Stat Mech 2013, Chung & Lu Ann Comb 2002)
here we generate the same marginal in and out-degree distributions
"""

def degdist(L1,L2,g1,g2,p,rho,N):
        
    import numpy as np    
    import scipy.special as spec
    import scipy.stats as stats
    
    d = np.arange(0,N)
    Pd = np.zeros(d.shape)
    Pd[0:L1+1] = d[0:L1+1]**g1
    Pd[L1+1:L2] = d[L1+1:L2]**g2
    
    ### make the degree distribution continuous at lower cutoff L1
    C1 = Pd[L1+1]/Pd[L1]
    Pd[0:L1] = C1*Pd[0:L1]
    
    ### normalize integral of the distribution
    Pin = Pd / np.sum(Pd)
    Pout = Pin 
    
    ### sample in  and out-degrees from bivariate gaussian copula 
    U_uniform = np.random.rand(N,2)
    
    ### evaluate inverse cdf of standard normal at the samples
    U_icdf = .5*(1+spec.erf(U_uniform/(np.sqrt(2.))))
    
    ### evaluate the joint cdf of a multivariant normal with mean zero and covariance matrix Rho at those values
    U = np.zeros((N,2))
    
    for i in range(N):
        er, val, inf = stats.mvn.mvndst(-10*np.ones((2,1)),U_icdf[i,:],np.zeros((2,1)),rho)
        U[i,:] += val
        
    # marginal cumulative distributions
    Cin = np.cumsum(Pin)
    Cout = np.cumsum(Pout)
    
    Din = np.zeros((N,1))
    Dout = np.zeros((N,1))
    
    for i in range(N):
        
        in_candidate = d[abs(Cin-U[i,0])==min(abs(Cin-U[i,0]))]
        # if multiple degrees are candidates (can happen if cumulative degree distribution hits 1 before d = N-1), pick one candidate degree
        Din[i] = in_candidate[0]
        
        out_candidate = d[abs(Cout-U[i,1])==min(abs(Cout-U[i,1]))]
        Dout[i] = out_candidate[0]

    ### sample connections with likelihoods proportional to in-degree of target and out-degree of source
    pref = np.dot(Din, Dout.T)
    pref = pref*(p/np.mean(pref,axis=(0,1)))

    adj = np.random.rand(N,N)       
    adj[adj>pref] = 0
    adj[adj!=0] = 1
    
    adj[range(N),range(N)] = 0
    
    return adj
        