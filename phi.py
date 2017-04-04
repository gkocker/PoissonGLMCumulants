# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:08:02 2015

@author: gabeo

define activation function for Poisson model
"""
import numpy as np


#
# #  softmax - smooth linear rectifier
#
# ''' here, log (1+e^(x)) '''
# def phi(g,gain):
#
#     gain = float(gain)
#     return gain*np.log(1+np.exp(g/gain))
#
# def phi_prime(g,gain):
#
#     gain = float(gain)
#     return 1. / (np.exp(-g/gain) + 1)
#
# def phi_prime2(g,gain):
#
#     gain = float(gain)
#     return np.exp(-g/gain) / (gain*(1+np.exp(-g/gain))**2)


'''here, exponential'''
def phi(g,gain):

    thresh = 0.
    r_out = gain*np.exp(g-thresh)

    return r_out

def phi_prime(g,gain):

    thresh = 0.
    phi_pr = gain*np.exp(g-thresh)

    return phi_pr

def phi_prime2(g,gain):

    '''
    second derivative of phi wrt input
    '''

    thresh = 0.
    phi_pr2 = gain * np.exp(g-thresh)

    return phi_pr2


def phi_pop(g,gain):

    '''

    :param g: 2d (E, I)
    :param gain: 2d(E, I)
    :return:
    '''

    thresh = 0.
    r_out = gain*np.exp(g-thresh)

    # r_out = gain*(g_calc**2)

    return r_out


# ''' here, half-wave rectified quadratic '''
# def phi(g,gain):
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc<thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = gain*(g_calc**2)
#
#     return r_out
#
# def phi_prime(g,gain):
#
#     g_calc = g*1.
#     thresh = 0.
#     ind = np.where(g_calc<thresh)
#     g_calc[ind[0]] = 0.
#     phi_pr = gain*2.*g_calc
#
#     return phi_pr
#
# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     g_calc = g*1
#     thresh = 0.
#     ind = np.where(g_calc<thresh)
#     ind1 = np.where(g_calc>=thresh)
#     g_calc[ind[0]] = 0.
#     g_calc[ind1[0]] = 1.
#     phi_pr2 = gain*2*g_calc
#
#     return phi_pr2
#
#
# def phi_pop(g,gain):
#
#     '''
#
#     :param g: 2d (E, I)
#     :param gain: 2d(E, I)
#     :return:
#     '''
#
#
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[0] = gain*g_calc[0]**2
#     r_out[1] = gain*g_calc[1]**2
#     # r_out = gain*(g_calc**2)
#
#     return r_out

# # #
# ''' here, half-wave rectified linear '''

# def phi(g, gain):
#    '''
#    voltage-rate transfer
#    '''
#
#
#
#    g_calc = g*1
#
#    thresh = 0
#    ind = np.where(g_calc<thresh)[0]
#    g_calc[ind] = 0
#
#    r_out = gain*g_calc
#
#    return r_out
#
#
# def phi_prime(g,gain):
#
#
#
#    g_calc = g*1
#    ind = np.where(g_calc<0)[0]
#
#    phi_pr = gain*np.ones(np.shape(g))
#    phi_pr[ind] = 0
#
#    return phi_pr
#
#
# def phi_prime2(g,gain):
#
#    '''
#    second derivative of phi wrt input
#    '''
#
#    g_calc = g*1
#    thresh = 0.
#    ind = np.where(g_calc == thresh)
#    phi_pr2 = np.zeros(g.shape)
#    # phi_pr2[ind] = 1.
#
#    return phi_pr2


# ''' here, concave down'''
#
# def phi(g, gain):
#
#     
#
#     # g_calc = g*1.
#
#     thresh = 0.
#     # ind = np.where(g_calc<thresh)
#     # g_calc[ind[0]] = 0.
#     #
#     # r_out = gain/2.*(1 + np.tanh(g/gain))
#     r_out = gain * (np.tanh(g/gain))
#     r_out[r_out <= thresh] = 0.
#
#
#     # r_out = gain*(1 + np.exp(-g_calc/gain))**-1
#     #
#     # g_calc[g_calc < thresh] = thresh
#     # r_out = gain*np.sqrt(g_calc)
#
#     return r_out
#
#
# def phi_prime(g, gain):
#
#     
#
#     g_calc = g*1.
#     thresh = 0.
#     # ind = np.where(g_calc < thresh)[0]
#     # g_calc[ind] = 0.
#
#     # phi_pr = np.exp(g_calc/gain) * (1+np.exp(g_calc/gain))**-2
#
#     r = phi(g, gain)
#     phi_pr = gain*np.cosh(g/gain)**(-2)
#     phi_pr[r <= thresh] = 0.
#
#     # g_calc[g_calc < thresh] = thresh
#     # phi_pr = gain * .5 * g_calc**(-.5)
#
#     return phi_pr
#
#
# def phi_prime2(g, gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     
#
#     g_calc = g*1
#     thresh = 0.
#     # ind = np.where(g_calc < thresh)
#     # g_calc[ind[0]] = 0.
#     #
#
#     # phi_pr2 = -(np.exp(g_calc/gain)-1)*np.exp(g_calc/gain) / (gain*(np.exp(g_calc/gain)+1)**3)
#
#     r = phi(g, gain)
#     phi_pr2 = -2. * np.cosh(g_calc/gain)**(-3) * np.sinh(g_calc/gain)
#     phi_pr2[r <= thresh] = 0.
#
#     # g_calc[g_calc < thresh] = thresh
#     # phi_pr2 = gain * .5 * -.5 * g_calc**(-1.5)
#
#     return phi_pr2
#
#
# ''' here, linear for E and quadratic for I '''
#
# def phi(g,gain):
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[:par.Ne] = gain*g_calc[:par.Ne]**1
#     r_out[par.Ne:] = gain*g_calc[par.Ne:]**2
#     # r_out = gain*(g_calc**2)
#
#     return r_out
#
# def phi_prime(g,gain):
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr = np.zeros(g_calc.shape)
#     phi_pr[:par.Ne] = gain*np.ones(par.Ne)
#     phi_pr[par.Ne:] = gain*2.*g_calc[par.Ne:]
#     phi_pr[ind[0]] = 0.
#
#     return phi_pr
#
# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr2 = np.zeros(g_calc.shape)
#     phi_pr2[:par.Ne] = 0.
#     phi_pr2[par.Ne:] = gain*2.*np.ones(par.Ni)
#     phi_pr2[ind[0]] = 0.
#
#     return phi_pr2
#
#
# def phi_pop(g,gain):
#
#     '''
#
#     :param g: 2d (E, I)
#     :param gain: 2d(E, I)
#     :return:
#     '''
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[0] = gain*g_calc[0]**1
#     r_out[1] = gain*g_calc[1]**2
#     # r_out = gain*(g_calc**2)
#
#     return r_out

# #
# ''' here, quadratic for E and linear for I '''
#
# def phi(g,gain):
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[:par.Ne] = gain*g_calc[:par.Ne]**2
#     r_out[par.Ne:] = gain*g_calc[par.Ne:]**1
#     # r_out = gain*(g_calc**2)
#
#     return r_out
#
# def phi_prime(g,gain):
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr = np.zeros(g_calc.shape)
#     phi_pr[:par.Ne] = gain*2.*g_calc[:par.Ne]
#     phi_pr[par.Ne:] = gain*np.ones(par.Ni)
#     phi_pr[ind[0]] = 0.
#
#     return phi_pr
#
# def phi_prime2(g,gain):
#
#     '''
#     second derivative of phi wrt input
#     '''
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1
#     thresh = 0.
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     phi_pr2 = np.zeros(g_calc.shape)
#     phi_pr2[:par.Ne] = gain*2.*np.ones(par.Ne)
#     phi_pr2[par.Ne:] = 0.
#     phi_pr2[ind[0]] = 0.
#
#     return phi_pr2
#
#
# def phi_pop(g,gain):
#
#     '''
#
#     :param g: 2d (E, I)
#     :param gain: 2d(E, I)
#     :return:
#     '''
#
#     
#     import params
#     reload(params)
#     par = params.params()
#
#     g_calc = g*1.
#
#     thresh = 0.from theory import rates_ss
#     ind = np.where(g_calc < thresh)
#     g_calc[ind[0]] = 0.
#
#     r_out = np.zeros(g_calc.shape)
#     r_out[0] = gain*g_calc[0]**2
#     r_out[1] = gain*g_calc[1]**1
#     # r_out = gain*(g_calc**2)
#
#     return r_out
