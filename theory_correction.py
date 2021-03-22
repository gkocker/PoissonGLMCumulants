# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:01:41 2015

@author: gabeo
"""
import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import numpy as np
from phi import phi, phi_prime, phi_prime2
import params; reload(params)


def two_point_function_integral_pop_1loop(W, ind_pop=None):
    """
    inputs: weight matrix, indices of neurons to sum two-point over
    calculate one-loop correction for two-point function in fluctuation expansion
    this is for the two-point function of the summed spike train of neurons in ind_pop, at zero frequency
    """

    par = params.params()
    N = par.N
    gain = par.gain
    b = par.b

    if ind_pop == None:
        ind_pop = range(par.Ne)

    phi_r = rates_ss(W)
    Tmax = 100
    dt_ccg = 1.
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    w_dummy = np.arange(-wmax, wmax, dw) * math.pi
    Nw_dummy = w_dummy.size
    dw_dummy = dw*math.pi

    w = np.array([0.]) # for 0 frequency
    # w = w_dummy.copy()
    Nw = w.size

    g0 = np.dot(W, phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1 = np.diag(phi_1)

    phi_2 = phi_prime2(g0, gain)
    phi_2 = np.diag(phi_2)

    Fbarsum = np.zeros((Nw_dummy,), dtype=complex)
    Fbar2 = np.zeros((Nw_dummy, ), dtype=complex)
    Fbarsum2_int = np.zeros((N, len(ind_pop), Nw), dtype=complex)
    Fbarsum2_int_conj = np.zeros((N, len(ind_pop), Nw), dtype=complex)

    C2f = np.zeros((1,), dtype=complex)  # fourier transform of stationary cross-covariance matrix

    ### first compute the loop from one source vertex to one internal vertex via two internal edges

    Fbar = np.zeros((Nw_dummy, N), dtype=complex) # 2 internal edges and source vertex
    for o in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
        Fbar_tmp = np.dot( W, g_fun(w_dummy[o])*F1_tmp ) # (N x N)
        Fbar[o] = np.dot(Fbar_tmp * Fbar_tmp.conj(), phi_r) # (N), conservation of momentum w 0-frequency outgoing edge gives the conj()

    Fbar_int = np.sum(Fbar, axis=0) * dw_dummy # (N,)
    Fbar_int = np.dot(phi_2/2., Fbar_int) / (2*math.pi) # scale internal vertex by gain

    ### start the one-loop correction
    F1_w0_full = linear_response_fun(0., np.dot(phi_1, W), phi_r)
    F1_w0 = np.sum(F1_w0_full[ind_pop], axis=0) # N,
    C2f = np.dot(F1_w0, np.dot(Fbar_int, F1_w0.conj().T))

    ### the next corrections are a tree two-point but one leg is replaced by the one-loop rate diagram
    Fbar = np.zeros((Nw_dummy, N), dtype=complex) # 2 internal edges and source vertex

    for o in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
        Fbar_tmp = np.dot( W, g_fun(w_dummy[o])*F1_tmp ) # (N x N)
        Fbar[o] = np.dot(Fbar_tmp * Fbar_tmp.conj(), phi_1.dot(F1_w0_full.conj())).dot(phi_r*F1_w0.conj())  # (N), conservation of momentum w 0-frequency outgoing edge gives the conj()

    Fbar_int = np.sum(Fbar, axis=0) * dw_dummy # (N,)
    Fbar_int = np.dot(phi_2/2., Fbar_int) / (2*math.pi) # scale internal vertex by gain

    C2f += 2. * np.dot(F1_w0, Fbar_int) # factor of two from symmetry of diagram

    ### the next corrections are a tree two-point with the one-loop propagator appended to one leg
    Fbar = np.zeros((Nw_dummy, N, N), dtype=complex)
    for o in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
        Fbar_tmp = np.dot( W, g_fun(w_dummy[o])*F1_tmp )
        Fbar[o] = np.dot(Fbar_tmp * Fbar_tmp.conj(), phi_1.dot(F1_w0_full).dot(np.diag(phi_r)))

    Fbar_int = np.sum(Fbar, axis=0) * dw_dummy
    Fbar_int = np.dot(phi_2/2., Fbar_int) / (2*math.pi)

    C2f += 2. * F1_w0.dot(Fbar_int).dot(F1_w0.conj())

    ### the next corections are from Michael Kordovan's thesis.
    ### first, the "M6" and "M7" diagrams
    Fbar = np.zeros((Nw_dummy,), dtype=complex)
    for o in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
        Fbar_tmp = np.dot( W, g_fun(w_dummy[o])*F1_tmp )

        Fbar[o] = F1_w0.dot((Fbar_tmp.conj().dot(phi_1.dot(F1_w0)).dot(Fbar_tmp.conj()) * phi_2.dot(Fbar_tmp)).dot(phi_r))

    C2f += 2. * np.sum(Fbar)*dw_dummy / 2. / (2*math.pi) # 2* from symmetry, /2 from internal vertex

    ### M8 - this is the one-loop rate correction with a tree-level two point appended
    Fbar = np.zeros((Nw_dummy, N), dtype=complex)
    Fbar_0 = np.dot( W, g_fun(0.)*F1_w0_full)

    for o in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
        Fbar_tmp = np.dot( W, g_fun(w_dummy[o])*F1_tmp )

        Fbar[o] = np.dot(Fbar_0, np.diag(phi_2) / 2. * np.dot(Fbar_tmp * Fbar_tmp.conj(), phi_r) ) #k

    Fbar_int = np.sum(Fbar, axis=0)*dw_dummy / (2*math.pi)
    Fbar_int = np.diag(phi_1) * Fbar_int
    C2f += np.sum(F1_w0 * F1_w0.conj() * Fbar_int)

    ### we don't have M9 and M10 bc the gain function doesn't have a third derivative.
    ### M11 and M12 - the strategy is to sum from the simple leg (direct to external vert) through the diagram
    C2f_tmp = 0.
    for o4 in range(Nw_dummy):
        F4_tmp = linear_response_fun(w_dummy[o4], np.dot(phi_1, W), phi_r)
        Fbar4_tmp = np.dot( W, g_fun(w_dummy[o4])*F4_tmp )
        Fbar4_tmp = np.dot(phi_2/2., Fbar4_tmp.dot(F1_w0 * phi_r)) # at l and omega_4

        for o1 in range(Nw_dummy):
            F1_tmp = linear_response_fun(w_dummy[o1], np.dot(phi_1, W), phi_r)

            F14_tmp = linear_response_fun(w_dummy[o1] - w_dummy[o4], np.dot(phi_1, W), phi_r)
            Fbar14_tmp = np.dot(W, g_fun(w_dummy[o1] - w_dummy[o4]*F14_tmp))

            Fbar14_tmp = F1_tmp.dot(np.diag(Fbar4_tmp).dot(Fbar14_tmp)) # km
            Fbar14_tmp = np.dot(Fbar14_tmp * F1_tmp.conj(), phi_r) # other km and sum over m (leaves size N,)

            C2f_tmp += np.dot(F1_w0, Fbar14_tmp)

    C2f += 2. * C2f_tmp * dw_dummy * dw_dummy / (2*math.pi)**2 # 2 from symmetry

    ### M13 and M14
    C2f_tmp = 0.
    for o1 in range(Nw_dummy):
        F1_tmp = linear_response_fun(w_dummy[o1], np.dot(phi_1, W), phi_r)
        Fbar1_tmp = np.dot( W, g_fun(w_dummy[o1])*F1_tmp )

        Fbar_w2_int = 0.

        for o2 in range(Nw_dummy):

            F2_tmp = linear_response_fun(w_dummy[o2], np.dot(phi_1, W), phi_r)
            F12_tmp = linear_response_fun(w_dummy[o1] - w_dummy[o2], np.dot(phi_1, W), phi_r)

            Fbar2_tmp = np.dot( W, g_fun(w_dummy[o2])*F2_tmp )
            Fbar12_tmp = np.dot( W, g_fun(w_dummy[o1] - w_dummy[o2])*F12_tmp )

            Fbar_w2_int += np.diag(phi_2)/2. * np.dot(Fbar2_tmp * Fbar12_tmp, phi_r) # l

        Fbar_w2_int = np.dot(np.dot(phi_2/2., Fbar1_tmp), Fbar_w2_int) # sum over l, leave k

        C2f_tmp += F1_tmp.dot(np.diag(Fbar_w2_int)).dot(Fbar1_tmp.conj()).dot(np.diag(phi_r)).dot(F1_w0.conj())

    C2f += 2. * C2f_tmp * dw_dummy * dw_dummy / (2*math.pi)**2

    ### M15
    C2f_tmp = 0.

    for o1 in range(Nw_dummy):

        F1_tmp = linear_response_fun(w_dummy[o1], np.dot(phi_1, W), phi_r)
        Fbar1_tmp = np.dot( W, g_fun(w_dummy[o1])*F1_tmp )

        C2f_tmp += Fbar1_tmp.T.dot(np.diag(F1_w0) * phi_1/2.).dot(Fbar1_tmp.conj())

    C2f_tmp *= dw_dummy
    C2f_tmp = phi_r.dot(C2f_tmp * C2f_tmp).dot(phi_r)

    C2f += C2f_tmp / (2*math.pi)**2

    return C2f