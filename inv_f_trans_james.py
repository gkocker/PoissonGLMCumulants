# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:04:44 2015

@author: gabeo
"""

def inv_f_trans_james(w, fhat):

    import numpy as np
    import math

    Nw = w.size
        
    dur = 1. / ((w[1]-w[0])) * 2* math.pi
    dt = dur/Nw

    jlist = np.arange(Nw)    
    
    t = -dur/2 + dt*jlist
    
    gtilde =  fhat*np.exp(-math.pi*1J*jlist)
    
    ftilde = (np.fft.ifft((gtilde)))
    
    f = 1. /(dt) * np.exp(math.pi*1j*Nw/2.) * np.exp(-math.pi*1j*jlist) * ftilde        
     
    return t, f

def inv_f_trans_james_2d(w1, w2, fhat):
    
    Nw1 = w1.size
    Nw2 = w2.size    
        
    for o2 in range(Nw1):
        t2, fhat[:,o2] = inv_f_trans_james(w1, fhat[:,o2])    
        
    for o1 in range(Nw2):
        t1, fhat[o1,:] = inv_f_trans_james(w2, fhat[o1,:])

    
#    f = fhat * (t1[1]-t1[0])**2 * Nw1**2 * 2
    f = fhat    
    
    return t1, t2, f