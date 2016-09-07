
''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj
from degdist import degdist
from generate_W_lognormal import generate_W as gen_W

import sim_poisson
import numpy as np
import matplotlib.pyplot as plt
from correlation_functions import bin_pop_spiketrain
from phi import phi_pop
from theory import rates_ss

from scipy.ndimage.filters import gaussian_filter

import time
import os
import sys


def bisection_WC(left, right, rE, rI, weights, ref):
    '''
    bisection to find nullclines of 2d wilson-cowan equation
    :param left: left initial endpoint (same for both variables)
    :param right: right initial endpoint (same for both variables
    :param rE: excitatory rate at which to find root (I rate for which drI_dt = 0)
    :param rI: inhibitory rate at which to find root (E rate for which drE_dt = 0)
    :param weights: [wEE, wEI, wIE, wII]
    :param ref: refractory period
    :return: r_0: rate at which dr/dt = 0
    '''

    tol = 1e-6
    max_its = 100
    left0 = left*1.
    right0 = right*1.

    par = params.params()
    Ne = par.Ne
    Ni = par.Ni
    pEE = par.pEE
    pEI = par.pEI
    pIE = par.pIE
    pII = par.pII
    tau = par.tau
    b = par.b
    gain = par.gain

    weightEE = weights[0]
    weightEI = weights[1]
    weightIE = weights[2]
    weightII = weights[3]

    ''' bisection for rI, freezing rE '''
    it = 0
    while it < max_its:

        if np.abs(left-right) < tol:
            # print 'root for rE found within tolerance'
            break

        g = [weightEE * rE + weightEI * left + np.mean(b[:Ne]), weightIE * rE + weightII * left + np.mean(b[Ne:])]
        g = np.array(g)
        drI_left = -left + (1. - ref*left) * phi_pop(g, gain)[1]

        g = [weightEE * rE + weightEI * right + np.mean(b[:Ne]), weightIE * rE + weightII * right + np.mean(b[Ne:])]
        g = np.array(g)
        drI_right = -right + (1. - ref*right) * phi_pop(g, gain)[1]

        mid = (left+right)/2.
        g = [weightEE * rE + weightEI * mid + np.mean(b[:Ne]), weightIE * rE + weightII * mid + np.mean(b[Ne:])]
        g = np.array(g)
        drI_mid = -mid + (1. - ref*mid) * phi_pop(g, gain)[1]

        # if (~np.any(np.array([drI_left, drI_right]) < 0.) or ~np.any(np.array([drI_left, drI_right]) > 0.)):
        #     # raise Exception('need wider initial range for bisection of rI')
        #     mid = np.nan
        #     break

        if np.sign(drI_mid) != np.sign(drI_left):
            right = mid * 1.
        elif np.sign(drI_mid) != np.sign(drI_right):
            left = mid * 1.
        else:
            mid = np.nan
            break
        it += 1

    drI_null = mid*1.

    ''' bisection for rE, freezing rI '''
    left = left0*1.  # reset initial endpoints
    right = right0*1.
    it = 0
    while it < max_its:

        if np.abs(left - right) < tol:
        #     print 'root for rI found within tolerance'
            break

        g = [weightEE * left + weightEI * rI + np.mean(b[:Ne]), weightIE * left + weightII * rI + np.mean(b[Ne:])]
        g = np.array(g)
        drE_left = -left + (1. - ref*left) * phi_pop(g, gain)[0]

        g = [weightEE * right + weightEI * rI + np.mean(b[:Ne]), weightIE * right + weightII * rI + np.mean(b[Ne:])]
        g = np.array(g)
        drE_right = -right + (1. - ref*right) * phi_pop(g, gain)[0]

        mid = (left + right) / 2.
        g = [weightEE * mid + weightEI * rI + np.mean(b[:Ne]), weightIE * mid + weightII * rI + np.mean(b[Ne:])]
        g = np.array(g)
        drE_mid = -mid + (1. - ref*mid) * phi_pop(g, gain)[0]

        # if (~np.any(np.array([drE_left, drE_right]) < 0.) or ~np.any(np.array([drE_left, drE_right]) > 0.)):
            # raise Exception('need wider initial range for bisection of rE')
            # mid = np.nan
            # break

        if np.sign(drE_mid) != np.sign(drE_left):
            right = mid * 1.
        elif np.sign(drE_mid) != np.sign(drE_right):
            left = mid * 1.
        else:
            mid = np.nan
            break

        it += 1

    drE_null = mid

    return drE_null, drI_null


def main(syn_scaleE=5., syn_scaleI=2.236):

    ''' unpackage parameters '''

    par = params.params()
    Ne = par.Ne
    Ni = par.Ni
    N = par.N
    pEE = par.pEE
    pEI = par.pEI
    pIE = par.pIE
    pII = par.pII
    tau = par.tau
    b = par.b
    gain = par.gain
    weightEE = par.weightEE
    weightEI = par.weightEI
    weightIE = par.weightIE
    weightII = par.weightII

    ''' set save directory '''
    if sys.platform == 'darwin': save_dir = '/Users/gocker/Documents/projects/structure_driven_activity/1loop_Ne=200_quadraticforIonly_b=0.1_forall/'
    elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/1loop_Ne=200_quadraticforIonly_b=0.1_forall/'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ''' load or generate adjacency matrix '''
    # W0_path = os.path.join(save_dir, 'W0.npy')

    # if os.path.exists(W0_path):
    #     W0 = np.load(W0_path)
    # else:

    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)
    #    W0EE = degdist(int(np.floor(Ne/10.)), Ne, .2, -1., pEE, .5, Ne)
    #    W0[:Ne, :Ne] = W0EE

        # if Ne > 0: # make first neuron a readout
        #     W0[0, :] = 0
        #     W0[0, 1:Ne] = 1
        #     W0[:, 0] = 0

        # np.save(W0_path, W0)

    print('Ne=' + str(Ne) + ', Ni=' + str(Ni))
    W = W0*1.
    W[:Ne, :Ne] *= weightEE
    W[:Ne, Ne:] *= weightEI
    W[Ne:, :Ne] *= weightIE
    W[Ne:, Ne:] *= weightII
    W[:Ne, :] *= syn_scaleE
    W[Ne:, :] *= syn_scaleI
    # rates_th = rates_ss(W)
    # rE_mean = np.mean(rates_th[:Ne])
    # rI_mean = np.mean(rates_th[Ne:])

    weightEE *= syn_scaleE*pEE*Ne
    weightEI *= syn_scaleE*pEI*Ni
    weightIE *= syn_scaleI*pIE*Ne
    weightII *= syn_scaleI*pII*Ni

    Nvec = 20
    r_max = 2000. / 1000.  # sp/ms
    r_vec = np.linspace(0, r_max, Nvec)
    r_meshE, r_meshI = np.meshgrid(r_vec, r_vec)
    drE_dt = np.zeros((Nvec, Nvec))
    drI_dt = np.zeros((Nvec, Nvec))

    Nnull = 500
    r_null = np.linspace(0, r_max, Nnull)
    drE_null = np.zeros((Nnull))
    drE_null_pos = np.zeros((Nnull))  # branch not including zero
    drI_null = np.zeros((Nnull))
    drI_null_pos = np.zeros((Nnull))  # branch not including zero

    ref = 1.

    ''' compute flow'''
    print 'computing flow'
    for i in range(Nvec):
        for j in range(Nvec):
            ''' compute flow '''
            rE = r_vec[j]
            rI = r_vec[i]
            g = [weightEE*rE + weightEI*rI + np.mean(b[:Ne]), weightIE*rE + weightII*rI + np.mean(b[Ne:])]
            g = np.array(g)
            drE_dt[i, j] = -rE + (1. - ref*rE) * phi_pop(g, gain)[0]
            drI_dt[i, j] = -rI + (1. - ref*rI) * phi_pop(g, gain)[1]

    fig, ax = plt.subplots(1, figsize=(4, 3))
    ax.quiver(r_meshE, r_meshI, drE_dt, drI_dt, angles='xy')
    # ax.quiver(r_meshE, r_meshI, drE_dt / np.sqrt(drE_dt**2 + drI_dt**2), drI_dt / np.sqrt(drE_dt**2 + drI_dt**2))

    ''' compute nullclines'''
    print 'computing nullclines'
    weights = np.array([weightEE, weightEI, weightIE, weightII])
    for i in range(Nnull):
        # drE_null_i, drI_null_i = bisection_WC(0., 1e-6, r_null[i], r_null[i], weights, ref)  # find branches including zero
        # drE_null[i] = drE_null_i
        # drI_null[i] = drI_null_i

        drE_null_i, drI_null_i = bisection_WC(1e-6, r_max, r_null[i], r_null[i], weights, ref)  # find branches including zero
        drE_null_pos[i] = drE_null_i
        drI_null_pos[i] = drI_null_i

    # ax.plot(drE_null, r_null, 'b', drE_null_pos, r_null, 'b', linewidth=2)
    ax.plot(r_null, drI_null, 'r', r_null, drI_null_pos, 'r', linewidth=2)

    ''' when E input is below threshold'''
    ax.plot(np.zeros(len(r_null[r_null > b[0]/(-weightEI)])), r_null[r_null > b[0]/(-weightEI)], 'b', linewidth=2)

    ''' when E input is above threshold '''
    det = (1./gain+(b[0]-syn_scaleE-2*syn_scaleE*r_null))**2 - 4*syn_scaleE*(2*syn_scaleE*r_null-b[0])
    mask = det >= 0.
    drE_null1 = (-(1./gain+(b[0]-syn_scaleE-2*syn_scaleE*r_null[mask])) + np.sqrt(det[mask])) /(2.*syn_scaleE)
    drE_null2 = (-(1./gain+(b[0]-syn_scaleE-2*syn_scaleE*r_null[mask])) - np.sqrt(det[mask])) /(2.*syn_scaleE)
    ax.plot(drE_null1, r_null[mask], 'b', drE_null2, r_null[mask], 'b', linewidth=2)

    ax.set_xlabel(r'$r_E$')
    ax.set_ylabel(r'$r_I$')
    # ax.set_title(r'$(r_E^0, r_I^0) = ($'+str(rE_mean)+', '+str(rI_mean)+')')
    ax.set_title(r'$W_{EE} = $'+str(weightEE/Ne/pEE))
    ax.set_xlim((-.001, r_max))
    ax.set_ylim((-.001, r_max))



    print 'running sims'
    tstop = 1000 * tau
    dt = .02*tau
    trans = 5*tau
    spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
    spk_Epop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, range(Ne)) / Ne
    spk_Ipop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, range(Ne, N)) / Ni

    r_Epop = gaussian_filter(spk_Epop, sigma=20)
    r_Ipop = gaussian_filter(spk_Ipop, sigma=20)
    ax.plot(r_Epop, r_Ipop, 'b', linewidth=0.2)

    print r'$W_{EE}=$'+str(weightEE/Ne/pEE)
    plt.show(fig)


if __name__ == '__main__':

    syn_scaleE = 100000
    syn_scaleI = np.sqrt(syn_scaleE)
    main(syn_scaleE, syn_scaleI)
