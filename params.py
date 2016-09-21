# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:04:07 2015

@author: gabeo

define a class to hold parameters
"""

import numpy as np

class params:
    def __init__(self):
        self.Ne = 200
        self.Ni = 40  # 40

        self.pEE = .2
        self.pEI = .5
        self.pIE = .5
        self.pII = .5
        if self.Ne > 0:
            self.WmaxE = 1./(self.Ne*self.pEE)
            self.weightEE = self.WmaxE  # have a WmaxE parameter from plasticity
            self.weightIE = 1./(self.Ne*self.pIE)
        else:
            self.weightEE = 0
            self.weightIE = 0
        if self.Ni > 0:
            self.weightEI = -2. / (self.Ni * self.pEI) # -2
            self.weightII = -2. / (self.Ni * self.pII) # -2
        else:
            self.weightEI = 0
            self.weightII = 0

        self.N_ff = 0  # 50 for coding fig
        self.Ne += self.N_ff
        self.N = self.Ne+self.Ni

        self.tau = 10  # time constant
        self.gain = .1 # gain of threshold-linear input-output function
        self.b = np.zeros((self.N,))

        # self.b[self.N_ff:] = 0.1

        self.b[self.N_ff:self.Ne] = 0.1  # if E cells have threshold-linear transfer
        self.b[self.Ne:self.N] = 0.15  # if I cells have threshold-quadratic nonlinearity

        self.b[:self.N_ff/2] = 0.1  # stim 2, nonlinear for coding fig
        # self.b[self.N_ff/2:self.N_ff] = 0.1  # stim 1, nonlinear for coding fig

        # self.b[:self.N_ff/2] = 0.1**2  # stim 2, linear for coding fig
        # self.b[self.N_ff/2:self.N_ff] = 0.1**2  # stim 1, linear for coding fig

        #  triplet plasticity parameters
        # self.b = .1*np.ones((self.N,)) # external input
        self.A3plus = 6.5e-3
        self.A2minus = 7.1e-3
        self.tauplus = 17 # msec
        self.tauminus = 34 # msec
        self.taux = 101 # msec
        self.tauy = 114 # msec
        self.eta = 1*1e-2 # learning rate parameter