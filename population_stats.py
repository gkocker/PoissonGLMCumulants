# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 13:04:01 2016

@author: gabeo
compute populaton spike train, spike counts and distribution thereof

"""

def bin_spiketrain(spktimes,dt,dt_ccg,tstop,trans,ind_exclude):
    
    import numpy as np
    
    ### figure out indices of where to put spikes from spike times
    ### first round spike times to dt, then divide
    ### then do corrfun = np.correlate(spk1,spk2,mode='valid'), return corrfun
    
    # find spike times
    ind_t = np.where(spktimes[:,1]!=ind_exclude)
    t_spk = spktimes[ind_t,0]-trans
    t_spk = np.transpose(t_spk)
    Nspk = np.size(ind_t)
    
    # covert times to indices
    ind_spk = np.floor(t_spk/dt_ccg)
    ind_spk = np.reshape(ind_spk,(Nspk,))
    ind_spk = ind_spk.astype(int)
    
    Nt = int((tstop)/dt_ccg)    
    
    spk = np.zeros((Nt,))
    spk[ind_spk] = 1
    
    return spk_pop

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