# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:01:48 2016

@author: gabeo

Figure: activity and discriminability as approach instability
"""

''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj
from degdist import degdist
from generate_W_lognormal import generate_W as gen_W

import sim_poisson
import numpy as np
import matplotlib.pyplot as plt
from raster import raster
from correlation_functions import auto_covariance_pop
from correlation_functions import bin_pop_spiketrain
from correlation_functions import cross_covariance_spk
from phi import phi_prime
from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier_pop
from theory import two_point_function_fourier_pop_1loop
from theory import stability_matrix_1loop

import time
import os
import sys

start_time = time.time()

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

Ntrials = 1
plot_raster = False

dt = .002 * tau
trans = 5. * tau

if plot_raster:
    tstop = 600. * tau + trans
    Ncalc = 3
else:
    tstop = 6000. * tau + trans
    # tstop = 10*tau
    Ncalc = 40  # 3 for rasters

window = tstop
Tmax = 8 * tau
dt_ccg = 1
lags = np.arange(-Tmax, Tmax, dt_ccg)

''' output saving '''
rE_av_theory = np.zeros((Ncalc, Ntrials))
rE_av_sim = np.zeros((Ncalc, Ntrials))
rI_av_sim = np.zeros((Ncalc, Ntrials))

r_readout_theory = np.zeros((Ncalc, Ntrials))
r_readout_sim = np.zeros((Ncalc, Ntrials))
rE_av_1loop = np.zeros((Ncalc, Ntrials))
r_readout_1loop = np.zeros((Ncalc, Ntrials))

if Ni > 0:
    rI_av_theory = np.zeros((Ncalc, Ntrials))
    rI_av_sim = np.zeros((Ncalc, Ntrials))
    rI_av_1loop = np.zeros((Ncalc, Ntrials))

spec_rad = np.zeros(Ncalc, )
spec_rad_1loop = np.zeros(Ncalc, )
two_point_integral_theory = np.zeros(Ncalc, )
two_point_integral_1loop = np.zeros(Ncalc, )
two_point_integral_sim = np.zeros((Ncalc, Ntrials))
two_point_readout_theory = np.zeros((Ncalc, Ntrials))
two_point_readout_sim = np.zeros((Ncalc, Ntrials, lags.size))
two_point_pop_sim = np.zeros((Ncalc, Ntrials, lags.size))

if plot_raster:
    # syn_scale = np.array((0., 1., 45.)) # for quadratic, was 75
    # syn_scale = np.array((0., 1., 12.)) # for linear
    syn_scale = np.array((0, 75.))  # for exponential

else:
    # syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
    # syn_scale = np.linspace(0., 60., Ncalc) # for linear
    syn_scale = np.linspace(0., 95., Ncalc) # for exponential, unbalanced


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gocker/Documents/projects/field_theory_spiking/1loop_Ne=200_exponential_transfer_lognormalWEE_nonER/'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/1loop_Ne=200_exponential_transfer_lognormalWEE_nonER/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

''' load or generate adjacency matrix '''
W0_path = os.path.join(save_dir, 'W0.npy')

if os.path.exists(W0_path):

    W0 = np.load(W0_path)
else:

    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)
    # W0EE = degdist(int(np.floor(Ne/10.)), Ne, .1, -.5, pEE, .5, Ne)
    W0EE = degdist(0, Ne, .01, -1., pEE, .8, Ne)
    W0[:Ne, :Ne] = W0EE

    # if Ne > 0: # make first neuron a readout
    #     W0[0, :] = 0
    #     W0[0, 1:Ne] = 1
    #     W0[:, 0] = 0

    np.save(W0_path, W0)


print('Ne=' + str(Ne) + ', Ni=' + str(Ni))

Nstable = Ncalc

W_store = np.zeros((N, N, Ncalc))


if not plot_raster:
    print 'computing theory'
    for nn in range(Ncalc):

        print 'progress %: ', float(nn)/float(Ncalc)*100

        ### generate scaled weight matrix from frozen connectivity realization
        # W = W0 * 1.
        # if Ne > 0:
        #     W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        #     W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]
        #
        # if Ni > 0:
        #     W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        #     W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

        # W *= syn_scale[nn]
        if syn_scale[nn] > 0.:
            W = gen_W(W0, Ne, syn_scale[nn]*weightEE, syn_scale[nn]*weightEE, syn_scale[nn]*weightEI, syn_scale[nn]*weightIE, syn_scale[nn]*weightII)
        else:
            W = np.zeros((N,N))

        W_store[:, :, nn] = W*1.
        r_th = rates_ss(W)
        rE_av_theory[nn] = np.mean(r_th[1:Ne]).real
        r_readout_theory[nn] = r_th[0].real

        r_th_11oop = rates_1loop(W)
        rE_av_1loop[nn] = np.mean(r_th_11oop[1:Ne]).real
        r_readout_1loop[nn] = r_th_11oop[0].real

        if Ni > 0:
           rI_av_1loop[nn] = np.mean(r_th_11oop[Ne:]).real
           rI_av_theory[nn] = np.mean(r_th[Ne:]).real

        g = np.dot(W, r_th) + b
        w = 0.
        stab_mat_mft = np.dot(np.diag(phi_prime(g, gain)), W)
        stab_mat_1loop = stability_matrix_1loop(w, W, r_th)
        spec_rad[nn] = max(abs(np.linalg.eigvals(stab_mat_mft)))
        spec_rad_1loop[nn] = max(abs(np.linalg.eigvals(stab_mat_mft + stab_mat_1loop)))

        if spec_rad[nn] >= 1.:
            print 'mft unstable at wEE = ', weightEE
            Nstable = nn
            break

        two_point_integral_theory[nn] = np.real(two_point_function_fourier_pop(W, range(Ne))[0])
        two_point_integral_1loop[nn] = np.real(two_point_function_fourier_pop_1loop(W, range(Ne))[0])


print 'running sims'
if plot_raster:
    calc_range = range(Nstable)
else:
    calc_range = range(0, Nstable, 2)

for nn in calc_range:

    print 'progress %: ', float(nn) / float(Ncalc) * 100

    ### generate scaled weight matrix from frozen connectivity realization
    # W = W0 * 1.
    # if Ne > 0:
    #     W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    #     W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]
    #
    # if Ni > 0:
    #     W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    #     W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    # W *= syn_scale[nn]
    # if syn_scale[nn] > 0.:
    #     W = gen_W(W0, Ne, syn_scale[nn] * weightEE, syn_scale[nn] * weightEE, syn_scale[nn] * weightEI,
    #               syn_scale[nn] * weightIE, syn_scale[nn] * weightII)
    # else:
    #     W = np.zeros((N, N))

    # except:
   #     print 'mft unstable, % complete = ' + str(float(nn) / float(Ncalc) * 100)

    W = W_store[:, :, nn]

    for nt in range(Ntrials):
        spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
   
        ind_include = range(1, Ne)
        spk_Epop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, ind_include)
        spk_readout = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, [0])
        rE_av_sim[nn, nt] = np.sum(spk_Epop) / np.amax(spktimes[:, 0]) / float(len(ind_include))
        r_readout_sim[nn, nt] = np.sum(spk_readout) / float(tstop)
   
        two_point_readout_sim[nn, nt, :] += cross_covariance_spk(spktimes, spktimes.shape[0], 0, 0, dt, lags, tau,
                                                                  tstop, trans)

        if Ni > 0:
            ind_include = range(Ne, N)
            spk_Ipop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, ind_include)
            rI_av_sim[nn, nt] = np.sum(spk_Ipop) / float(tstop) / float(len(ind_include))
   
        two_point_pop_sim[nn, nt, :] = auto_covariance_pop(spktimes, range(0, N), spktimes.shape[0], dt, lags, tau,
                                                            tstop, trans)
        two_point_integral_sim[nn, nt] = np.sum(two_point_pop_sim[nn, :]) * 1
   
    if plot_raster and nn == 1 or plot_raster and nn == Ncalc-1:
        print 'plotting raster'
        savefile = os.path.join(save_dir, 'raster_scale=' + str(syn_scale[nn]) + '.pdf')
        spktimes[:,0] -= trans
        raster(spktimes, spktimes.shape[0], tstop, savefile, size=(1.875, 1.875))

        print 'plotting population two-point function'



end_time = time.time()
print end_time - start_time
#
syn_scale *= weightEE

meanEEweight = np.mean(W_store[:Ne, :Ne, :], axis=(0,1))

# ''' Plot figures '''
if not plot_raster:

    size = (5., .8)

    fig, ax = plt.subplots(1, figsize=size)
    ax.plot(meanEEweight[:Nstable], rE_av_theory[:Nstable],  'k', label='Tree level', linewidth=2)
    if not 'linear' in save_dir: ax.plot(meanEEweight[:Nstable], rE_av_theory[:Nstable]+rE_av_1loop[:Nstable], 'r' , label='One loop', linewidth=2);
    ax.plot(meanEEweight[0:Ncalc:2], rE_av_sim[0:Ncalc:2], 'ko', label='Sim')
    ax.set_ylabel('E Population Rate (sp/ms)')
    ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
    ax.set_xlim((0, np.amax(meanEEweight)))
    ax.set_ylim((0, rE_av_theory[0]*2))
    # ax.legend(loc=0, fontsize=10)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Arial')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
        item.set_fontname('Arial')

    savefile = os.path.join(save_dir, 'rE_vs_weight.pdf')
    fig.savefig(savefile)
    plt.show(fig)
    plt.close(fig)

    fig, ax = plt.subplots(1, figsize=size)
    ax.plot(meanEEweight[:Nstable], r_readout_theory[:Nstable], 'k', label='Tree level', linewidth=2)
    if not 'linear' in save_dir: ax.plot(meanEEweight[:Nstable], r_readout_theory[:Nstable] + r_readout_1loop[:Nstable], 'r', label='One loop', linewidth=2);
    ax.plot(meanEEweight[0:Ncalc:2], r_readout_sim[0:Ncalc:2], 'ko', label='Sim')
    ax.set_ylabel('Readout Rate (sp/ms)')
    ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
    ax.set_xlim((0, np.amax(meanEEweight)))
    if 'quadratic' in save_dir:
        ax.set_ylim((0, r_readout_theory[0]*20))
    elif 'linear' in save_dir:
        ax.set_ylim((0, np.amax(r_readout_theory)*1.5))
    # ax.legend(loc=0, fontsize=10)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Arial')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
        item.set_fontname('Arial')

    savefile = os.path.join(save_dir, 'r_readout_vs_weight.pdf')
    fig.savefig(savefile)
    plt.show(fig)
    plt.close(fig)

    fig, ax = plt.subplots(1, figsize=size)
    ax.plot(meanEEweight[:Nstable], rI_av_theory[:Nstable], 'k', label='Tree level', linewidth=2)
    if not 'linear' in save_dir: ax.plot(syn_scale[:Nstable], rI_av_theory[:Nstable] + rI_av_1loop[:Nstable], 'r', label='One loop',
                                         linewidth=2);
    ax.plot(meanEEweight[0:Ncalc:2], rI_av_sim[0:Ncalc:2], 'ko', label='Sim')
    ax.set_ylabel('I Population Rate (sp/ms)')
    ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
    ax.set_xlim((0, np.amax(meanEEweight)))
    ax.set_ylim((0, rI_av_theory[0]*2))
    # ax.legend(loc=0, fontsize=10)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Arial')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
        item.set_fontname('Arial')

    savefile = os.path.join(save_dir, 'rI_vs_weight.pdf')
    fig.savefig(savefile)
    plt.show(fig)
    plt.close(fig)

    fig, ax = plt.subplots(1, figsize=size)
    ax.plot(meanEEweight[:Nstable], two_point_integral_theory[:Nstable], 'k', label='Tree level', linewidth=2)
    if not 'linear' in save_dir: ax.plot(meanEEweight[:Nstable], two_point_integral_theory[:Nstable] + two_point_integral_1loop[:Nstable], 'r', label='One loop',
                                         linewidth=2);
    ax.plot(meanEEweight[0:Ncalc:2], two_point_integral_sim[0:Ncalc:2], 'ko', label='Sim')
    ax.set_ylabel('E Pop. Spike Train Variance (sp/ms)^2')
    ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
    ax.set_xlim((0, np.amax(meanEEweight)))
    if 'quadratic' in save_dir:
        ax.set_ylim((0, two_point_integral_theory[0]*30))
    elif 'linear' in save_dir:
        ax.set_ylim((0, np.amax(two_point_integral_theory*1.5)))
    # ax.legend(loc=0, fontsize=10)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Arial')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
        item.set_fontname('Arial')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    savefile = os.path.join(save_dir, 'var_Epop_vs_weight.pdf')
    fig.savefig(savefile)
    plt.show(fig)
    plt.close(fig)

    fig, ax = plt.subplots(1, figsize=size)
    ax.plot(meanEEweight[:Nstable], spec_rad[:Nstable], 'k', label='Tree level', linewidth=2)
    if not 'linear' in save_dir: 
        ax.plot(meanEEweight[:Nstable], spec_rad_1loop[:Nstable],
                                         'r', label='One loop', linewidth=2)
                                         
        ax.plot(meanEEweight, np.ones(syn_scale.shape), 'k--')
                                         
    ax.set_ylabel('Spectral radius of mean field theory')
    ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
    ax.set_xlim((0, np.amax(meanEEweight)))
    ax.set_ylim((0, 1.5))
    # ax.legend(loc=0, fontsize=10)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(10)
        item.set_fontname('Arial')
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
        item.set_fontname('Arial')

    savefile = os.path.join(save_dir, 'stability_vs_weight.pdf')
    fig.savefig(savefile)
    plt.show(fig)
    plt.close(fig)


    ''' save data '''

    savefile = os.path.join(save_dir, 'rE_av_sim.npy')
    np.save(savefile, rE_av_sim)

    savefile = os.path.join(save_dir, 'r_readout_sim.npy')
    np.save(savefile, r_readout_sim)

    savefile = os.path.join(save_dir, 'rI_av_sim.npy')
    np.save(savefile, rI_av_sim)

    savefile = os.path.join(save_dir, 'two_point_pop_sim.npy')
    np.save(savefile, two_point_pop_sim)

    savefile = os.path.join(save_dir, 'two_point_readout_sim.npy')
    np.save(savefile, two_point_readout_sim)

    savefile = os.path.join(save_dir, 'two_point_integral_sim.npy')
    np.save(savefile, two_point_integral_sim)

    savefile = os.path.join(save_dir, 'rE_av_theory.npy')
    np.save(savefile, rE_av_theory)

    savefile = os.path.join(save_dir, 'r_readout_theory.npy')
    np.save(savefile, r_readout_theory)

    savefile = os.path.join(save_dir, 'rI_av_theory.npy')
    np.save(savefile, rI_av_theory)

    savefile = os.path.join(save_dir, 'rE_av_1loop.npy')
    np.save(savefile, rE_av_1loop)

    savefile = os.path.join(save_dir, 'r_readout_1loop.npy')
    np.save(savefile, r_readout_1loop)

    savefile = os.path.join(save_dir, 'rI_av_1loop.npy')
    np.save(savefile, rI_av_1loop)

    savefile = os.path.join(save_dir, 'two_point_integral_theory.npy')
    np.save(savefile, two_point_integral_theory)

    savefile = os.path.join(save_dir, 'two_point_integral_1loop.npy')
    np.save(savefile, two_point_integral_1loop)

print syn_scale
print syn_scale/weightEE