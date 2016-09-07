# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:04:44 2015

@author: gabeo
"""

def inv_f_trans(w, fhat):
    
    import numpy as np
    import math
    
    Nw = w.size
    
    one_over_dur = w[1]-w[0]
    dur = 1/one_over_dur
    
    t = np.linspace(-dur,dur,Nw)
    t = t*math.pi

    if w[0] < 0:
        fhat_arranged = np.fft.ifftshift(fhat)
    else: # only have positive frequency half 
        fhat_arranged = fhat

    f = np.fft.ifftshift(np.fft.ifft(fhat_arranged)) 
#    f = np.fft.fftshift(np.fft.ifft(fhat_arranged))
    
    f /= (t[1]-t[0])

    return t, f
    
def inv_f_trans_2d(w1, w2, fhat):

    Nw1 = w1.size
    Nw2 = w2.size    
    
    import numpy as np
    import math
#    
    one_over_dur = w1[1]-w1[0]
    dur = 1/one_over_dur
    
    t1 = np.linspace(-dur,dur,Nw1)
    t1 = t1*math.pi
    
    one_over_dur = w2[1]-w2[0]
    dur = 1/one_over_dur
    
    t2 = np.linspace(-dur,dur,Nw2)
    t2 = t2*math.pi

    if w1[0] < 0:
        fhat_arranged = np.fft.ifftshift(fhat)
    else: # only have positive frequency half
        fhat_arranged = fhat 
        
    f = np.fft.fftshift(np.fft.ifft2(fhat_arranged, axes=(0,1)))
    f = f / (t1[1]-t1[0])

#    for o2 in range(Nw1):
#        t2, fhat[:,o2] = inv_f_trans(w1, fhat[:,o2])
#    
#    for o1 in range(Nw2):
#        t1, fhat[o1,:] = inv_f_trans(w2, fhat[o1,:])
#    
#    f = fhat * (t1[1]-t1[0])
    
    return t1, t2, f