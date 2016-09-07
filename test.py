# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:55:38 2015

@author: gabeo

testinginverse fourier transforms


"""

import numpy as np
import matplotlib.pyplot as plt
from inv_f_trans import inv_f_trans
from inv_f_trans import inv_f_trans_2d
import math

dw = .1
wmax = 200
w = np.arange(-wmax, wmax, dw)
Nw = w.size

'''1-d '''
fhat = np.zeros((Nw,))
freq = 1.
 
fhat[np.where(np.abs(w-freq)==np.amin(np.abs(w-freq)))] = 1. / dw
fhat[np.where(np.abs(w+freq)==np.amin(np.abs(w+freq)))] = 1. / dw

t, f = inv_f_trans(w, fhat)
plt.figure(); plt.plot(t/math.pi,f)

fhat = np.ones((Nw,))
t, f = inv_f_trans(w, fhat)
plt.figure(); plt.plot(t/math.pi, f)

'''2-d'''
fhat = np.ones((Nw,Nw))
t1, t2, f = inv_f_trans_2d(w, w, fhat)
plt.figure(); plt.plot(t1, f[:,Nw/2])

fhat = np.zeros((Nw,Nw), dtype = 'complex128')
fhat[np.where(np.abs(w-freq)==np.amin(np.abs(w-freq))), Nw/2] = 1. / dw + 10j
t1, t2, f = inv_f_trans_2d(w, w, fhat)
plt.figure(); plt.plot(t1, f[:,Nw/2])


'''
test spike counts from long sim
'''

T = 200
R = 40000
tstop = R*T/2 + trans
import time
start_time = time.time()

spktimes, g_vec2 = sim_poisson(W, tstop, trans, dt)
spktimes[:, 0] -= trans

T_start = np.arange(0, tstop-trans, T/2)
# R = len(T_start)
r_sim = np.zeros((1, R, N))


for r, t in enumerate(T_start):
    ind_start = np.where(spktimes[:, 0] >= t)[0][0]
    ind_end = np.where(spktimes[:, 0] < t+T)[0][-1]
    for n in range(N):
        r_sim[nn, r, n] = sum(spktimes[ind_start:ind_end, 1] == n)
        # r_sim[nn, r, n] = sum((spktimes[:, 1] == n) & (spktimes[:, 0] >= t) & (spktimes[:, 0] < t+T))

end_time = time.time()
print end_time-start_time, ' seconds'