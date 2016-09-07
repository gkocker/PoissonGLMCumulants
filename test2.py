# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:14:49 2015

@author: gabeo
"""


import numpy as np
import matplotlib.pyplot as plt



wmax = 100
Nw = 10000
w = np.linspace(-wmax,wmax,Nw)

testf = -1/(1+1j*w)
testf_arranged = np.fft.fftshift(testf)

test = np.fft.ifft(testf_arranged)

figtest = plt.figure()
plt.plot(test)