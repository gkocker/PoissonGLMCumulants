# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:05:02 2015

@author: gabeo

compute covariance of spike trains,

""" 
    
def bin_spiketrain(spktimes,neuronind,dt,dt_ccg,tstop,trans):
    
    import numpy as np
    
    # find spike times
    ind_t = np.where(spktimes[:,1]==neuronind)
    t_spk = spktimes[ind_t,0]-trans
    t_spk = np.transpose(t_spk)
    Nspk = np.size(ind_t)
    
    # covert times to indices
    ind_spk = np.floor(t_spk/dt_ccg)
    ind_spk = np.reshape(ind_spk,(Nspk,))
    ind_spk = ind_spk.astype(int)
    
    Nt = int(tstop/dt_ccg)+1    
    
    spk = np.zeros((Nt,))
    spk[ind_spk] = 1
    
    return spk


def bin_pop_spiketrain(spktimes,dt,dt_ccg,tstop,trans,ind_include):
    
    import numpy as np
    
    Nt = int(tstop/dt_ccg)+1
    
    spk_pop = np.zeros((Nt,))    
    
    for i in ind_include:
        spk_temp = bin_spiketrain(spktimes, i, dt, dt_ccg, tstop, trans)
        spk_pop += spk_temp
    
    return spk_pop

def auto_covariance_pop(spktimes,ind_include,numspikes,dt,lags,tau,tstop,trans):

    import numpy as np
    import params; reload(params)
    par = params.params()
    N = par.N

    spk = bin_pop_spiketrain(spktimes,dt,1,tstop,trans,ind_include)
    spk = spk[int(trans/dt):] / float(len(ind_include))
    
    r = np.sum(spk)/(tstop-trans)
    spk -= r    
    
    ### compute cross-covariance for each time lag
    Nlags = lags.size
    Nt = spk.size
    auto_cov = np.zeros(Nlags,)
    
    for i in range(0,Nlags):
        nshift = int(lags[i]) # number of bins, and direction, to shift by
        if nshift >= 0: 
            auto_cov[i] += np.dot(spk[nshift:Nt:1],np.conj(spk[0:Nt-nshift:1]))
        else:
            nshift = -nshift
            auto_cov[i] += np.conj(np.dot(np.conj(spk[0:Nt-nshift:1]),spk[nshift:Nt:1]))
        ### unbiased estimate of cross-correlation
        auto_cov[i] = auto_cov[i]/(Nt-np.abs(lags[i]))
    
    return auto_cov

def count_dist(spk,dt_ccg,tstop,win):
    
    """
    compute vector of population spike counts from one trial in half-overlapping windows
    inputs:
        spk - population spike train
        dt_ccg - time bin
        tstop - total sim time (with transient already subtracted)
        win - counting window
    """

    import numpy as np

    tstop = tstop/dt_ccg
    win = win/dt_ccg
    
    Nwin = int(tstop/win*2)
    count = np.zeros((Nwin,))    
    
    for i in range(Nwin):
        count[i] = sum(spk[i*win/2:(i+2)*win/2])
        
    return count

def cross_covariance_spk(spktimes,numspikes,ind1,ind2,dt,lags,tau,tstop,trans):
    ### another method: create binary spike trains at inputed dt, compute correlation function with numpy.correlate
    
    import numpy as np
        
    spk1 = bin_spiketrain(spktimes,ind1,dt,1,tstop,trans)
    spk2 = bin_spiketrain(spktimes,ind2,dt,1,tstop,trans)
    
    spk1 = spk1[int(trans/dt):]    
    spk2 = spk2[int(trans/dt):] 
    
    r1 = np.sum(spk1)/(tstop-trans)
    r2 = np.sum(spk2)/(tstop-trans) 
    
    spk1 -= r1
    spk2 -= r2    
    
    ### compute cross-covariance for each time lag
    Nlags = lags.size
    Nt = spk1.size
    xcov = np.zeros(Nlags,)
    
    for i in range(0,Nlags):
        nshift = int(lags[i]) # number of bins, and direction, to shift by
        if nshift >= 0: 
            xcov[i] += np.dot(spk1[nshift:Nt:1],np.conj(spk2[0:Nt-nshift:1]))
        else:
            nshift = -nshift
            xcov[i] += np.conj(np.dot(np.conj(spk1[0:Nt-nshift:1]),spk2[nshift:Nt:1]))
        ### unbiased estimate of cross-correlation
        xcov[i] = xcov[i]/(Nt-np.abs(lags[i]))
    
    
    return xcov
    
def triplet_covariance_spk(spktimes,numspikes,ind1,ind2,ind3,dt,lags,tau,tstop,trans):
    '''
    compute three-point function for a range of time lags
    note: third cumulant is the third central moment, which is what we do directly here
    '''

    import numpy as np
    
    spk1 = bin_spiketrain(spktimes,ind1,dt,1,tstop,trans)
    spk2 = bin_spiketrain(spktimes,ind2,dt,1,tstop,trans)
    spk3 = bin_spiketrain(spktimes,ind3,dt,1,tstop,trans)
    
    spk1 = spk1[int(trans/dt):]    
    spk2 = spk2[int(trans/dt):]    
    spk3 = spk3[int(trans/dt):]    
    
    Nt = len(spk1)    
    
    r1 = np.sum(spk1)/(tstop-trans)
    r2 = np.sum(spk2)/(tstop-trans)
    r3 = np.sum(spk3)/(tstop-trans)
    
    spk1 -= r1
    spk2 -= r2
    spk3 -= r3    
    
    Nlags = lags.size
    Nt = spk1.size
    maxlag = -lags[0]
    xcov3 = np.zeros((Nlags,Nlags,))   
    
    for i in np.arange(-maxlag, maxlag+1):    
        if i >= 0:
            spk2_calc = np.concatenate((spk2[i:], np.zeros(i,)))
        else:
            spk2_calc = np.concatenate((np.zeros((-i,)), spk2[0:Nt+i]))
        for j in np.arange(-maxlag, maxlag+1):
            if j >= 0:
                spk3_calc = np.concatenate((spk3[j:], np.zeros((j,))))
            else:
                spk3_calc = np.concatenate((np.zeros((-j,)), spk3[0:Nt+j]))
            
            xcov3[i+maxlag, j+maxlag] = np.sum(spk1*spk2_calc*spk3_calc)/(Nt-np.abs(i))
    
    return xcov3
                    
def triplet_covariance_tot(spktimes,numspikes,ind1,ind2,ind3,dt,lags,tau,tstop,trans):
    
    '''compute integral of three-point function over s1 and s2 by evaluating bispectrum at 0,0 '''
    
    import numpy as np    
    import math
    
    dt_ccg = lags[1]-lags[0]    
    
    spk1 = bin_spiketrain(spktimes,ind1,dt,dt_ccg,tstop,trans)
    spk2 = bin_spiketrain(spktimes,ind2,dt,dt_ccg,tstop,trans)
    spk3 = bin_spiketrain(spktimes,ind3,dt,dt_ccg,tstop,trans)
    
    spk1 = spk1[int(trans/dt):]    
    spk2 = spk2[int(trans/dt):]     
    spk3 = spk3[int(trans/dt):] 

    r1 = np.sum(spk1)/(tstop-trans)
    r2 = np.sum(spk2)/(tstop-trans)
    r3 = np.sum(spk3)/(tstop-trans)
    
#    spk1 = spk1/dt
#    spk2 = spk2/dt
#    spk3 = spk3/dt        
    
    spk1 -= r1
    spk2 -= r2
    spk3 -= r3
##    
    spk1_f = np.fft.fft(spk1)
    spk2_f = np.fft.fft(spk2)
    spk3_f = np.fft.fft(spk3)
    
    cov3_tot = spk1_f[0].conj()*spk2_f[0]*spk3_f[0]*math.pi/1000
    
    return cov3_tot

def cross_covariance_tot(spktimes,numspikes,ind1,ind2,dt,lags,tau,tstop,trans):
    
    import numpy as np
    import math    
    from scipy import signal
    
    dt_ccg = lags[1]-lags[0]    
    
    spk1 = bin_spiketrain(spktimes,ind1,dt,dt_ccg,tstop,trans)
    spk2 = bin_spiketrain(spktimes,ind2,dt,dt_ccg,tstop,trans)
    
    spk1 = spk1[int(trans/dt):]    
    spk2 = spk2[int(trans/dt):]     
    
    r1 = np.sum(spk1)/(tstop-trans)
    r2 = np.sum(spk2)/(tstop-trans) 
    
    spk1 -= r1
    spk2 -= r2    
    
    spk1_f = np.fft.fft(spk1)
    spk2_f = np.fft.fft(spk2)
#    
#    xcov_tot = spk1_f[0].conj()*spk2_f[0] / ((tstop-trans)/dt) / 2 / 1000
    freq, cross_spec = signal.csd(spk1, spk2, fs=1./dt_ccg, window = 'hanning', scaling = 'density', return_onesided=False)
    xcov_tot = cross_spec[0]
    return xcov_tot
    
def cross_spectrum(spktimes,numspikes,ind1,ind2,dt,lags,tau,tstop,trans):
    import numpy as np
    import math   
    from scipy import signal
    
    dt_ccg = lags[1]-lags[0]    
    spk1 = bin_spiketrain(spktimes,ind1,dt,dt_ccg,tstop,trans)
    spk2 = bin_spiketrain(spktimes,ind2,dt,dt_ccg,tstop,trans)
    
    spk1 = spk1[int(trans/dt):]    
    spk2 = spk2[int(trans/dt):]     
    
    r1 = np.sum(spk1)/(tstop-trans)
    r2 = np.sum(spk2)/(tstop-trans) 
    
    spk1 -= r1
    spk2 -= r2    
    
    freq, cross_spec = signal.csd(spk1, spk2, fs=1./dt_ccg, window = 'bartlett', nperseg = 256, scaling = 'density', return_onesided=False)
    
    return freq, cross_spec/(dt_ccg**2)