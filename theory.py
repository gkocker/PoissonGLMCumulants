# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:01:41 2015

@author: gabeo
"""

import math
import numpy as np
from phi import phi, phi_prime, phi_prime2
import params; reload(params)

def rates_ss(W): # set inputs here

    '''
    :param W: weight matrix
    :return: steady-state rates with transfer functions and cellular/synaptic parameters defined in params.py and phi.py
    '''
    
    par = params.params()
    b = par.b
    gain = par.gain
    tau = par.tau
    N = par.N

    dt = .02*tau
    Tmax = int(50*tau / dt)
    a = 1. / tau
    a2 = a**2

    r = np.zeros(N)
    s_dummy = np.zeros(N)
    s = np.zeros(N)

    r_vec = np.zeros((N, Tmax))
    for i in range(Tmax):
        s_dummy += dt*(-2*a*s_dummy - a2*s) + r*a2*dt
        s += dt * s_dummy

        g = W.dot(s) + b
        r = phi(g, gain)
        r_vec[:, i] = r

    return r


def rates_1loop(W):
    
    """
    inputs: weight matrix
    calculate one-loop correction for steady-state firing rates in fluctuation expansion
    """    

    import params; reload(params)
    from phi import phi_prime
    from phi import phi_prime2
        
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    phi_r = rates_ss(W)
    Tmax = 100
    dt_ccg = 1.
    wmax = 1. / dt_ccg
    dw = 1. / Tmax
    
    w = np.arange(-wmax, wmax, dw) * math.pi
    dw = dw*math.pi
    Nw = w.size

    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1 = np.diag(phi_1)   
    
    phi_2 = phi_prime2(g0, gain)
    Fbarsum = np.zeros((N, Nw), dtype=complex)

    for o in range(Nw): # first compute Fbar over dummy frequency
        Fbar1 = np.dot(g_fun(w[o])*W, linear_response_fun(w[o], np.dot(phi_1,W), phi_r))
        Fbarsum[:, o] = np.dot(Fbar1*Fbar1.conj(), phi_r)  # sum over first inner vertex
    
    Fbarsum_int = np.sum(Fbarsum, axis=1)*dw  # integrate over dummy frequency
    
    F1 = linear_response_fun(0., np.dot(phi_1,W), phi_r)
    
    r_1loop = np.dot(F1, .5*phi_2*Fbarsum_int) / ((2*math.pi)**1) # sum over second inner vertex
    return r_1loop
        
        
def two_point_function_fourier(W):
    
    """
    calculate tree-level prediction for two-point function in fluctuation expansion
    inputs: weight matrix
    leave out factors of two pi that come with delta functions because we have an implicit inverse fourier transform from one of the frequencies that is left out because we are dealing only with stationary processes
    """
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b

    phi_r = rates_ss(W)
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1 = np.diag(phi_1)   
    W = np.dot(phi_1, W)

    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    #w = np.arange(-wmax, wmax, dw) * math.pi
    w = np.arange(0., 1, 1)
    Nw = w.size

    C2f = np.zeros((N,N,Nw), dtype=complex)  # fourier transform of stationary cross-covariance matrix

    phi_r = np.diag(phi_r)
    for o in range(Nw):
        F1 = linear_response_fun(w[o], W, phi_r)
        C2f[:, :, o] = F1.dot(phi_r).dot(F1.conj().T)
#        
    return C2f, w


def two_point_function_fourier_1loop(W):

    """
    inputs: weight matrix
    calculate one-loop correction for two-point function in fluctuation expansion
    """
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    phi_r = rates_ss(W)
    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax
    
    w_dummy = np.arange(-wmax, wmax, dw) * math.pi
    w = np.arange(0, 1, 1)
    Nw_dummy = w_dummy.size
    Nw = w.size

    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1 = np.diag(phi_1)   
    
    phi_2 = phi_prime2(g0,gain)
    phi_2 = np.diag(phi_2)
    
    Fbarsum = np.zeros((N, Nw_dummy), dtype=complex)
    Fbarsum2_int = np.zeros((N, N, Nw), dtype=complex)
    Fbarsum2_int_conj = np.zeros((N, N, Nw), dtype=complex)

    for o1 in range(Nw):
        F1 = linear_response_fun(w[o1], np.dot(phi_1, W), phi_r)
        for o in range(Nw_dummy): # first compute Fbar over dummy frequency
            Fbar1 = np.dot(W, g_fun(w_dummy[o])*linear_response_fun(w_dummy[o], np.dot(phi_1,W), phi_r))
            Fbarsum[:, o] = np.dot(Fbar1*Fbar1.conj(), phi_r)

            Fbar2 = np.dot(W, g_fun(w[o1] - w_dummy[o]) * linear_response_fun(w[o1] - w_dummy[o], np.dot(phi_1, W), phi_r))
            Fbar2conj = np.dot(W, g_fun(-w[o1] - w_dummy[o]) * linear_response_fun(-w[o1] - w_dummy[o], np.dot(phi_1, W), phi_r))
            Fbarsum2_int[:, :, o1] += np.dot(Fbar1 * Fbar2, np.diag(phi_r)).dot(F1.T.conj())*dw
            Fbarsum2_int_conj[:, :, o1] += np.dot(Fbar1 * Fbar2conj, np.diag(phi_r)).dot(F1.T)*dw


    Fbarsum_int = np.sum(Fbarsum, axis=1)*dw # integrate over dummy frequency
    Fbarsum_int = np.diag(Fbarsum_int)

    C2f = np.zeros((N, N, Nw), dtype=complex) # fourier transform of stationary cross-covariance matrix

    for o in range(Nw):
        F1 = linear_response_fun(w[o], np.dot(phi_1, W), phi_r)
        F1loop= linear_response_1loop(w[o], np.dot(phi_1,W), phi_r)
        C2f[:, :, o] = np.dot(np.dot(phi_2/2. * Fbarsum_int, F1), F1.conj().T) / (2*math.pi)**0
        C2f[:, :, o] += np.dot(F1, F1loop.conj()) / (2*math.pi)**0
        C2f[:, :, o] += np.dot(F1.conj(), F1loop) / (2*math.pi)**0
        C2f[:, :, o] += np.dot(np.dot(F1, phi_2/2.), Fbarsum2_int[:, :, o])
        C2f[:, :, o] += np.dot(np.dot(F1.conj(), phi_2 / 2.), Fbarsum2_int_conj[:, :, o])

    return C2f, w


def g_fun(w):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    tau = par.tau

    taud = 0.
    
    g = np.exp(-1j*w*taud) / ((1+1j*w*tau)**2)  # alpha function

    return g

    
def linear_response_fun(w, W, phi_r):
    
    par = params.params()
    N = par.N

    Gamma = g_fun(w)*W  # W has already been multiplied by the gain of the rate function
    Delta = np.linalg.inv(np.eye(N) - Gamma)
    
    return Delta


def linear_response_1loop(w, W, phi_r):

    '''
    calculate one-loop correction to the propagator around mean-field theory
    :param w: frequency
    :param W: weight matrix, weighted by postsynaptic gain
    :param phi_r: firing rates
    :return: propagator matrix
    '''


    par = params.params()

    b = par.b
    gain = par.gain
    N = par.N

    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    w_calc = np.arange(-wmax, wmax, dw) * math.pi
    dw *= math.pi
    Nw = w_calc.size

    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1_diag = np.diag(phi_1)

    phi_2 = phi_prime2(g0,gain)
    phi_2_diag = np.diag(phi_2)

    F1 = linear_response_fun(w, np.dot(phi_1_diag,W), phi_r)
    Fbar = np.dot(g_fun(w)*W, F1)

    Fbar_int = np.zeros((N, N), dtype='complex128')

    for o in range(Nw):
        Fbar1 = np.dot(g_fun(w_calc[o])*W, linear_response_fun(w_calc[o], np.dot(phi_1_diag,W), phi_r))
        Fbar2 = np.dot(g_fun(w-w_calc[0])*W, linear_response_fun(w-w_calc[o], np.dot(phi_1_diag,W), phi_r))
        Fbar_int += np.dot(Fbar1*Fbar2, np.dot(phi_1_diag, Fbar))*dw

    linear_response_1loop = np.dot(np.dot(F1, phi_2_diag/2.), Fbar_int) / (2*math.pi**1)

    return linear_response_1loop


def stability_matrix_1loop(w, W, phi_r):
    '''
    one-loop correction to linear stability matrix around mean-field
    :param w: frequency
    :param W: weight matrix
    :param phi_r: firing rates
    :return:
    '''

    par = params.params()

    b = par.b
    gain = par.gain
    N = par.N

    Tmax = 200.
    dt_ccg = 1.
    wmax = 1. / dt_ccg
    dw = 1. / Tmax
    
    w_calc = np.arange(-wmax, wmax, dw) * math.pi
    dw *= math.pi
    Nw = w_calc.size  
        
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1_diag = np.diag(phi_1)       
    
    phi_2 = phi_prime2(g0,gain)
    phi_2_diag = np.diag(phi_2)
    
    F1 = linear_response_fun(w, np.dot(phi_1_diag, W), phi_r)
    Fbar_int = np.zeros((N, N), dtype='complex128')

    for o in range(Nw):
        Fbar1 = np.dot(g_fun(w_calc[o])*W, linear_response_fun(w_calc[o], np.dot(phi_1_diag, W), phi_r))
        Fbar2 = np.dot(g_fun(w-w_calc[o])*W, linear_response_fun(w-w_calc[o], np.dot(phi_1_diag, W), phi_r))
        Fbar_int += np.dot(Fbar1*Fbar2, np.dot(phi_1_diag, g_fun(w)*W))*dw  # remove incoming internal edge

    stab_1loop = np.dot(phi_2_diag/2., Fbar_int) / (2*math.pi**1)

    return stab_1loop


def two_point_function_fourier_pop(W, ind_pop):
    '''
    calculate tree-level two-point function of the summed spike train of neurons in ind_pop
    :param W: weight matrix
    :param ind_pop: neurons to sum over
    :return: fourier transform of the population two-point function
    '''
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    phi_r = rates_ss(W)
    phi_r_diag = np.diag(phi_r)
    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax
    
    # w = np.arange(-wmax, wmax, dw) * math.pi
    w = np.arange(0., 1, 1)
    Nw = w.size  
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1 = np.diag(phi_1)   
    W = np.dot(phi_1, W)
    
    C2f = np.zeros((Nw,), dtype=complex)  # fourier transform of stationary cross-covariance matrix
    
    for o in range(Nw):
        F1 = linear_response_fun(w[o], W, phi_r)
        F1_pop = np.sum(F1[ind_pop, :], axis=0) / float(len(ind_pop))
        C2f[o] = np.dot(F1_pop, np.dot(F1_pop.conj().T, phi_r_diag))
    
    return C2f


def two_point_function_fourier_pop_1loop(W, ind_pop):
    """
    inputs: weight matrix, indices of neurons to sum over
    calculate one-loop correction for two-point function in fluctuation expansion
    this is for the two-point function of the summed spike train of neurons in ind_pop
    """
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    phi_r = rates_ss(W)
    Tmax = 100
    dt_ccg = 1.
    wmax = 1. / dt_ccg
    dw = 1. / Tmax
    
    w_dummy = np.arange(-wmax, wmax, dw) * math.pi
    Nw_dummy = w_dummy.size  
    dw_dummy = dw*math.pi
    
    w = np.arange(0., 1, 1)
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

    C2f = np.zeros((Nw,), dtype=complex)  # fourier transform of stationary cross-covariance matrix

    for o1 in range(Nw):
        F1 = linear_response_fun(w[o1], np.dot(phi_1, W), phi_r)

        for o in range(Nw_dummy):  # first compute Fbar over dummy frequency
            F1 = linear_response_fun(w_dummy[o], np.dot(phi_1, W), phi_r)
            F1_pop = np.sum(F1, axis=0) / float(N)
            Fbar1 = np.dot(W, g_fun(w_dummy[o])*F1_pop)
            Fbarsum[o] = np.dot(Fbar1*Fbar1.conj(), phi_r)

    Fbarsum_int = np.sum(Fbarsum)*dw_dummy  # integrate over dummy frequency

    for o1 in range(Nw):
        F1 = linear_response_fun(w[o1], np.dot(phi_1, W), phi_r)
        F1ind = F1[ind_pop, :]
        for o in range(Nw_dummy): # first compute Fbar over dummy frequency
            Fbar2 = np.dot(W, g_fun(w[o1] - w_dummy[o]) * linear_response_fun(w[o1] - w_dummy[o], np.dot(phi_1, W), phi_r))
            Fbar2conj = np.dot(W, g_fun(-w[o1] - w_dummy[o]) * linear_response_fun(-w[o1] - w_dummy[o], np.dot(phi_1, W), phi_r))
            Fbarsum2_int[:, :, o1] += np.dot(np.dot(Fbar1 * Fbar2, np.diag(phi_r)), F1ind.T.conj())*dw
            Fbarsum2_int_conj[:, :, o1] += np.dot(np.dot(Fbar1 * Fbar2conj, np.diag(phi_r)), F1ind.T)*dw

    Fbarsum2_int = np.sum(Fbarsum2_int, axis=1)
    Fbarsum2_int_conj = np.sum(Fbarsum2_int_conj, axis=1)

    for o in range(Nw):
        F1 = linear_response_fun(w[o], np.dot(phi_1, W), phi_r)
        F1_pop = np.sum(F1[ind_pop, :], axis=0) / float(len(ind_pop))
        F1loop = linear_response_1loop(w[0], np.dot(phi_1, W), phi_r)
        F1loop_pop = np.sum(F1loop[ind_pop], axis=0) / float(len(ind_pop))

        C2f[o] = np.dot(np.dot(phi_2/2. * Fbarsum_int, F1_pop), F1_pop.conj().T) / (2*math.pi)**1
        C2f[o] += np.dot(F1_pop, F1loop_pop.conj()) / (2*math.pi)**0
        C2f[o] += np.dot(F1_pop.conj(), F1loop_pop) / (2*math.pi)**0
        C2f[o] += np.dot(F1_pop, Fbarsum2_int[:, o])
        C2f[o] += np.dot(F1_pop.conj(), Fbarsum2_int_conj[:, o])

    return C2f


def two_point_function_fourier_freq(W, w):
    """
    calculate tree-level prediction for two-point function in fluctuation expansion
    inputs: weight matrix
    leave out factors of two pi that come with delta functions because we have an implicit inverse fourier transform from one of the frequencies that is left out because we are dealing only with stationary processes
    """

    par = params.params()
    N = par.N
    gain = par.gain
    b = par.b

    phi_r = rates_ss(W)

    g0 = np.dot(W, phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1 = np.diag(phi_1)
    W = np.dot(phi_1, W)

    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    # w = np.arange(-wmax, wmax, dw) * math.pi
    # w = np.arange(0., 1, 1)
    Nw = len(w)

    C2f = np.zeros((N, N, Nw), dtype=complex)  # fourier transform of stationary cross-covariance matrix

    for o in range(Nw):
        F1 = linear_response_fun(w[o], W, phi_r)
        C2f[:, :, o] = np.dot(phi_r * F1, F1.conj().T)
    #
    return C2f


def two_point_function_fourier_lags(W, lags):
    """
    inputs: weight matrix, maximum lag, bin width for cross-correlation
    """

    import numpy as np
    import math
    import params;
    reload(params)
    from phi import phi_prime

    par = params.params()
    N = par.N
    gain = par.gain
    b = par.b

    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')

    phi_r = rates_ss(W)

    wmax = 1. / ((lags[1] - lags[0])) * 2 * math.pi
    dw = wmax / lags.size

    w = -wmax / 2. + dw * np.arange(lags.size)
    #    w[np.where(np.abs(w)<1e-6)] = 1e-6

    #    w = np.arange(-wmax, wmax, dw) * math.pi
    #    w, trash = inv_f_trans_james(lags, np.zeros(lags.shape))
    Nw = w.size

    g0 = np.dot(W, phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1 = np.diag(phi_1)
    W = np.dot(phi_1, W)

    #    phi_r = np.diag(phi_r)

    C2f = np.zeros((N, N, Nw), dtype=complex)  # fourier transform of stationary cross-covariance matrix

    for o in range(Nw):
        #        C2f[:,:,o] = np.dot(phi_r*Delta[:,:,o], Delta[:,:,o].conj().T)
        F1 = linear_response_fun(w[o], W, phi_r)
        F2 = linear_response_fun(w[o], W, phi_r)
        C2f[:, :, o] = np.dot(phi_r * F1, F2.conj().T)
    #
    return C2f, w


def two_point_function_lags(W, lags):
    import numpy as np
    import params;
    reload(params)

    par = params.params()
    N = par.N

    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')

    C2f, w = two_point_function_fourier_lags(W, lags)

    C2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(C2f, axes=2), axis=2), axes=2) / (lags[1] - lags[0])
    for i in range(N):
        for j in range(N):
            C2[i, j, :] -= C2[i, j, 0]

    C2 = C2.real

    return C2
