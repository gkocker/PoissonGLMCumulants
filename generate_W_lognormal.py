# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:16:21 2015

@author: gabeo

generate weight matrix with lognormal E-E weights
+
dependency: numpy
"""
import numpy as np


def generate_W(W0, Ne, mean_weightEE, var_weightEE, weightEI, weightIE, weightII):


    W = W0.copy()

    # print weightEE
    # ind = np.where(W0[:Ne, :Ne] == 1)
    #
    # for i in ind:
    #     W[i] *= np.random.lognormal(weightEE, var_weightEE)

    mu_EE = np.log((mean_weightEE**2) / np.sqrt(var_weightEE + mean_weightEE**2))
    sig2_EE = np.log(1. + (var_weightEE / (mean_weightEE**2)))
    sig_EE = np.sqrt(sig2_EE)

    W[:Ne, :Ne] *= np.random.lognormal(mu_EE, sig_EE, size=(Ne, Ne))

    ''' delta-distributed EI, IE, II blocks'''

    W[:Ne, Ne:] *= weightEI # to E from I
    W[Ne:, :Ne] *= weightIE # to I from E
    W[Ne:, Ne:] *= weightII # to I from I

    return W