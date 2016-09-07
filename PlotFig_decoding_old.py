'''
Plot figure: mutual information between activity and stimulus for different versions of the theory
params for stim 1: b[:N_ff] = 2*regular
params for sti 2: b[N_ff:2*N_ff] = 2*regular
'''


''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj


import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import multivariate_normal as mv_norm
from scipy.stats import norm
from scipy.stats import sem

from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier
from theory import two_point_function_fourier_1loop
from theory_mutual_information import mutual_inf
from theory_mutual_information import entropy_gaussian
import sim_poisson

import time
import os
import sys


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gocker/Documents/projects/structure_driven_activity/encoding/'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/encoding/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

''' unpackage parameters '''
par = params.params()
Ne = par.Ne
Ni = par.Ni
N_ff = par.N_ff
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

''' load or generate adjacency matrix '''
W0_path = os.path.join(save_dir, 'W0_2layer.npy')

if os.path.exists(W0_path):
    W0 = np.load(os.path.join(save_dir, 'W0_2layer.npy'))
else:
    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)

    if Ne > 0 and N_ff > 0: # make first 2 groups of 50 neurons a projection layer
        W0_ff = np.random.rand(Ne, 2*N_ff)
        W0_ff[W0_ff > .5] = 0.
        W0_ff[W0_ff != 0] = 1.
        W0[:Ne, :2*N_ff] = 0
        W0[:Ne, :2 * N_ff] += W0_ff
        W0[:2*N_ff, :] = 0  # no inputs to layer 1
        W0[Ne:, :2*N_ff] = 0  # layer 1 doesn't target I cells

        # W0[Ne, :] = 0  # make last E neuron a readout neuron
        W0[:, Ne] = 0
        W0[Ne, Ne:] = 0

    savefile = os.path.join(save_dir, 'W0_2layer.npy')
    np.save(savefile, W0)


''' simulation parameters '''
tstop = 100.*tau
trans = 5.*tau
dt = .02*tau

''' plot log odds of stim given activity as synaptic strength scales '''
Ncalc = 10
N_code = Ne - 2*N_ff

r_tree_stim1 = np.zeros((Ncalc, N_code))
r_1loop_stim1 = np.zeros((Ncalc, N_code))
cov_tree_stim1 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim1 = np.zeros((Ncalc, N_code, N_code))

r_tree_stim2 = np.zeros((Ncalc, N_code))
r_1loop_stim2 = np.zeros((Ncalc, N_code))
cov_tree_stim2 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim2 = np.zeros((Ncalc, N_code, N_code))

R = 10
p_stim1 = 0.5
p_stim2 = 0.5

r_sim_stim1 = np.zeros((Ncalc, N_code, R))
log_odds_tree = np.zeros((Ncalc, R))
log_odds_1loop = np.zeros((Ncalc, R))
log_odds_lin_noise = np.zeros((Ncalc, R))
log_odds_quad_noise = np.zeros((Ncalc, R))
log_odds_tree_readout = np.zeros((Ncalc, R))
log_odds_1loop_readout = np.zeros((Ncalc, R))
log_odds_lin_noise_readout  = np.zeros((Ncalc, R))
log_odds_quad_noise_readout = np.zeros((Ncalc, R))

''' calculate theory for stim 1 '''
print 'ranging over synaptic weights for linear network, stim 1'

phi_set = raw_input("Enter 'True' after checking that transfer is threshold-linear in phi.py")
b_set = raw_input("Enter 'True' after setting b for stim 1 (linear) in params.py")
reload(params)

syn_scale = np.linspace(0., 12., Ncalc)  # for linear

for nn in range(Ncalc):

    print 'progress %: ', float(nn)/float(Ncalc)*100

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale[nn]
    r_tree_stim1[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1
    r_1loop_stim1[nn, :] = r_tree_stim1[nn, :] + rates_1loop(W)[2*N_ff:2*N_ff+N_code].real*1

    cov_tree_stim1[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1
    cov_1loop, w = two_point_function_fourier_1loop(W)
    cov_1loop_stim1[nn, :, :] = cov_tree_stim1[nn, :, :] + np.real(cov_1loop[2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1

    for r in range(R):
        spktimes, g_vec1 = sim_poisson.sim_poisson(W, tstop, trans, dt)
        spktimes[:, 0] -= trans
        r_temp = np.zeros(N)
        for n in range(N):
            r_temp[n] = sum(spktimes[:, 1] == n) / float(tstop - trans)

        r_sim_stim1[nn, :, r] = r_temp[2*N_ff : 2*N_ff + N_code] * 1

''' calculate theory for stim 2 and mutual inf '''
b_set = raw_input("Enter 'True' after setting b for stim 2 (linear) in params.py")
reload(params)

for nn in range(Ncalc):

    print 'progress %: ', float(nn)/float(Ncalc)*100

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale[nn]
    r_tree_stim2[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1
    r_1loop = rates_1loop(W)
    r_1loop_stim2[nn, :] = r_tree_stim2[nn, :] + r_1loop[2*N_ff:2*N_ff+N_code].real*1

    cov_tree_stim2[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1
    cov_1loop, w = two_point_function_fourier_1loop(W)
    cov_1loop_stim2[nn, :, :] = cov_tree_stim2[nn, :, :] + np.real(cov_1loop[2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1

    for r in range(R):

        ''' log odds of stim 1 vs stim 2 given E population activity '''
        ind = np.where(r_tree_stim1[nn, :] == 0.)[0]
        ind = list(ind)

        r_sim_stim1_ma = np.delete(r_sim_stim1[nn, :, r], ind, axis=0)
        r_tree_stim1_ma = np.delete(r_tree_stim1[nn, :], ind, axis=0)
        r_1loop_stim1_ma = np.delete(r_1loop_stim1[nn, :], ind, axis=0)
        cov_tree_stim1_ma = np.delete(cov_tree_stim1[nn, :, :], ind, axis=0)
        cov_tree_stim1_ma = np.delete(cov_tree_stim1_ma, ind, axis=1)
        cov_1loop_stim1_ma = np.delete(cov_1loop_stim1[nn, :, :], ind, axis=0)
        cov_1loop_stim1_ma = np.delete(cov_1loop_stim1_ma, ind, axis=1)

        p_r1_stim1_tree = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim1_ma, np.diag(np.diag(cov_tree_stim1_ma)))
        p_r1_stim1_1loop = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim1_ma, cov_tree_stim1_ma)
        p_r1_stim1_lin_noise = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim1_ma, cov_tree_stim1_ma)
        p_r1_stim1_quad_noise = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim1_ma, cov_1loop_stim1_ma)

        ind = np.where(r_tree_stim2[nn, :] == 0.)[0]
        ind = list(ind)

        r_sim_stim1_ma = np.delete(r_sim_stim1[nn, :, r], ind, axis=0)
        r_tree_stim2_ma = np.delete(r_tree_stim2[nn, :], ind, axis=0)
        r_1loop_stim2_ma = np.delete(r_1loop_stim2[nn, :], ind, axis=0)
        cov_tree_stim2_ma = np.delete(cov_tree_stim2[nn, :, :], ind, axis=0)
        cov_tree_stim2_ma = np.delete(cov_tree_stim2_ma, ind, axis=1)
        cov_1loop_stim2_ma = np.delete(cov_1loop_stim2[nn, :, :], ind, axis=0)
        cov_1loop_stim2_ma = np.delete(cov_1loop_stim2_ma, ind, axis=1)

        p_r1_stim2_tree = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim2_ma, np.diag(np.diag(cov_tree_stim2_ma)))
        p_r1_stim2_1loop = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim2_ma, cov_tree_stim2_ma)
        p_r1_stim2_lin_noise = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim2_ma, cov_tree_stim2_ma)
        p_r1_stim2_quad_noise = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim2_ma, cov_1loop_stim2_ma)

        p_stim1_r1_tree = p_r1_stim1_tree + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim1_r1_lin_noise = p_r1_stim1_lin_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim1_r1_1loop = p_r1_stim1_1loop + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim1_r1_quad_noise = p_r1_stim1_quad_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        p_stim2_r1_tree = p_r1_stim2_tree + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim2_r1_lin_noise = p_r1_stim2_lin_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim2_r1_1loop = p_r1_stim2_1loop + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim2_r1_quad_noise = p_r1_stim2_quad_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        log_odds_tree[nn, r] = p_stim1_r1_tree - p_stim2_r1_tree
        log_odds_1loop[nn, r] = p_stim1_r1_1loop - p_stim2_r1_1loop
        log_odds_lin_noise[nn, r] = p_stim1_r1_lin_noise - p_stim2_r1_lin_noise
        log_odds_quad_noise[nn, r] = p_stim1_r1_quad_noise - p_stim2_r1_quad_noise

        ''' log odds of stim 1 vs stim 2 given readout activity '''

        p_r1_stim1_tree = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_1loop = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_lin_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_quad_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim1[nn, -1], cov_1loop_stim1[nn, -1, -1])

        p_r1_stim2_tree = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_1loop = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_lin_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_quad_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim2[nn, -1], cov_1loop_stim2[nn, -1, -1])

        p_stim1_r1_tree = p_r1_stim1_tree + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim1_r1_lin_noise = p_r1_stim1_lin_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim1_r1_1loop = p_r1_stim1_1loop + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim1_r1_quad_noise = p_r1_stim1_quad_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        p_stim2_r1_tree = p_r1_stim2_tree + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim2_r1_lin_noise = p_r1_stim2_lin_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim2_r1_1loop = p_r1_stim2_1loop + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim2_r1_quad_noise = p_r1_stim2_quad_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        log_odds_tree_readout[nn, r] = p_stim1_r1_tree - p_stim2_r1_tree
        log_odds_1loop_readout[nn, r] = p_stim1_r1_1loop - p_stim2_r1_1loop
        log_odds_lin_noise_readout[nn, r] = p_stim1_r1_lin_noise - p_stim2_r1_lin_noise
        log_odds_quad_noise_readout[nn, r] = p_stim1_r1_quad_noise - p_stim2_r1_quad_noise


''' set up subplot grid '''
fig1 = plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(2, 1)
gs1.update(top=.95, bottom=.05, hspace=0.1, left=0.05, right=0.25)
ax1 = plt.subplot(gs1[0, 0])
ax2 = plt.subplot(gs1[1, 0])

gs2 = gridspec.GridSpec(2, 1)
gs2.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax3 = plt.subplot(gs2[0, 0])
ax4 = plt.subplot(gs2[1, 0])

'''
plot transfer functions
'''
x = np.arange(-1, 1, .01)
y = gain*x
y[np.where(x<0)[0]] = 0
ax1.plot(x, y, 'k', linewidth=2)
ax1.set_xlabel('Membrane voltage', fontsize=12)
ax1.set_ylabel('Firing rate', fontsize=12)
ax1.set_xlim((-1, 1))
ax1.set_ylim((-.001, .1))
ax1.set_xticks([0])
ax1.set_yticks([])

y = gain*x**2
y[np.where(x<0)[0]] = 0
ax2.plot(x, y, 'k', linewidth=2)
ax2.set_xlabel('Membrane voltage', fontsize=12)
ax2.set_ylabel('Firing rate', fontsize=12)
ax2.set_xlim((-1, 1))
ax2.set_ylim((-.001, .1))
ax2.set_xticks([0])
ax2.set_yticks([])

''' plot linear network '''
Nstab = 38
# ax3.errorbar(syn_scale*weightEE, log_odds_tree.mean(axis=1), yerr=sem(log_odds_tree, axis=1), label='Tree')
# ax3.errorbar(syn_scale*weightEE, log_odds_1loop.mean(axis=1), yerr=sem(log_odds_1loop, axis=1), label='1 Loop')
ax3.errorbar(syn_scale*weightEE, log_odds_lin_noise.mean(axis=1), yerr=sem(log_odds_lin_noise, axis=1), label='Linear Noise')
ax3.errorbar(syn_scale*weightEE, log_odds_quad_noise.mean(axis=1), yerr=sem(log_odds_quad_noise, axis=1), label='Quadratic Noise')
# ax3.plot([syn_scale[Nstab]*weightEE, syn_scale[Nstab]*weightEE], [.8*np.amin(inf_tree[:, 0]), 1.2*np.amax(inf_tree[:, 0])], 'k')


''' set up subplot grid '''
fig2 = plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(2, 1)
gs1.update(top=.95, bottom=.05, hspace=0.1, left=0.05, right=0.25)
ax5 = plt.subplot(gs1[0, 0])
ax6 = plt.subplot(gs1[1, 0])

gs2 = gridspec.GridSpec(2, 1)
gs2.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax7 = plt.subplot(gs2[0, 0])
ax8 = plt.subplot(gs2[1, 0])

'''
plot transfer functions
'''
x = np.arange(-1, 1, .01)
y = gain*x
y[np.where(x<0)[0]] = 0
ax5.plot(x, y, 'k', linewidth=2)
ax5.set_xlabel('Membrane voltage', fontsize=12)
ax5.set_ylabel('Firing rate', fontsize=12)
ax5.set_xlim((-1, 1))
ax5.set_ylim((-.001, .1))
ax5.set_xticks([0])
ax5.set_yticks([])

y = gain*x**2
y[np.where(x<0)[0]] = 0
ax6.plot(x, y, 'k', linewidth=2)
ax6.set_xlabel('Membrane voltage', fontsize=12)
ax6.set_ylabel('Firing rate', fontsize=12)
ax6.set_xlim((-1, 1))
ax6.set_ylim((-.001, .1))
ax6.set_xticks([0])
ax6.set_yticks([])

''' plot linear network '''
Nstab = 38
# ax7.errorbar(syn_scale*weightEE, log_odds_tree_readout.mean(axis=1), yerr=sem(log_odds_tree_readout, axis=1), label='Tree')
# ax7.errorbar(syn_scale*weightEE, log_odds_1loop_readout.mean(axis=1), yerr=sem(log_odds_1loop_readout, axis=1), label='1 Loop')
ax7.errorbar(syn_scale*weightEE, log_odds_lin_noise_readout.mean(axis=1), yerr=sem(log_odds_lin_noise_readout, axis=1), label='Linear Noise')
ax7.errorbar(syn_scale*weightEE, log_odds_quad_noise_readout.mean(axis=1), yerr=sem(log_odds_quad_noise_readout, axis=1), label='Quadratic Noise')
# ax3.plot([syn_scale[Nstab]*weightEE, syn_scale[Nstab]*weightEE], [.8*np.amin(inf_tree[:, 0]), 1.2*np.amax(inf_tree[:, 0])], 'k')

''' set up subplot grid for readout plots '''
Ncalc_readout_plot = int(np.floor(Ncalc / 3.))

fig3 = plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(2, 1)
gs1.update(top=.95, bottom=.05, hspace=0.1, left=0.05, right=0.25)
ax9 = plt.subplot(gs1[0, 0])
ax10 = plt.subplot(gs1[1, 0])

gs2 = gridspec.GridSpec(2, 1)
gs2.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax11 = plt.subplot(gs2[0, 0])
ax12 = plt.subplot(gs2[1, 0])

'''
plot transfer functions
'''
x = np.arange(-1, 1, .01)
y = gain*x
y[np.where(x<0)[0]] = 0
ax9.plot(x, y, 'k', linewidth=2)
ax9.set_xlabel('Membrane voltage', fontsize=12)
ax9.set_ylabel('Firing rate', fontsize=12)
ax9.set_xlim((-1, 1))
ax9.set_ylim((-.001, .1))
ax9.set_xticks([0])
ax9.set_yticks([])

y = gain*x**2
y[np.where(x<0)[0]] = 0
ax10.plot(x, y, 'k', linewidth=2)
ax10.set_xlabel('Membrane voltage', fontsize=12)
ax10.set_ylabel('Firing rate', fontsize=12)
ax10.set_xlim((-1, 1))
ax10.set_ylim((-.001, .1))
ax10.set_xticks([0])
ax10.set_yticks([])

W = W0 * 1.
if Ne > 0:
    W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

if Ni > 0:
    W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

W *= syn_scale[Ncalc_readout_plot]
b_set = raw_input("Enter 'True' after setting b for stim 1 (linear) in params.py")

r_readout_tree_stim1 = rates_ss(W)[Ne] * 1
r_readout_1loop_stim1 = r_readout_tree_stim1 + rates_1loop(W)[Ne] * 1
sig_readout_tree_stim1 = np.real(two_point_function_fourier(W)[0])[Ne, Ne]*1
sig_readout_1loop_stim1 = sig_readout_tree_stim1 + np.real(two_point_function_fourier_1loop(W)[0])[Ne, Ne]*1

r_plot = np.arange(0., 1000., .1) / 1000.
ax11.plot(r_plot, norm.pdf(r_plot, r_readout_tree_stim1, sig_readout_tree_stim1), 'k')
ax11.plot(r_plot, norm.pdf(r_plot, r_readout_1loop_stim1.real, sig_readout_1loop_stim1.real), 'r')

b_set = raw_input("Enter 'True' after setting b for stim 2 (linear) in params.py")
r_readout_tree_stim2 = rates_ss(W)[Ne] * 1
r_readout_1loop_stim2 = r_readout_tree_stim2 + rates_1loop(W)[Ne] * 1
sig_readout_tree_stim2 = np.real(two_point_function_fourier(W)[0])[Ne, Ne]*1
sig_readout_1loop_stim2 = sig_readout_tree_stim2 + np.real(two_point_function_fourier_1loop(W)[0])[Ne, Ne]*1

ax11.plot(r_plot, norm.pdf(r_plot, r_readout_tree_stim2, sig_readout_tree_stim2), 'k--')
ax11.plot(r_plot, norm.pdf(r_plot, r_readout_1loop_stim2.real, sig_readout_1loop_stim2.real), 'r--')

print 'ranging over synaptic weights for nonlinear network, stim 1'

syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
phi_set = raw_input("Enter 'True' after checking that transfer is threshold-quadratic in phi.py")
b_set = raw_input("Enter 'True' after setting b for stim 1 (nonlinear) in params.py")
reload(params)

r_tree_stim1 = np.zeros((Ncalc, N_code))
r_1loop_stim1 = np.zeros((Ncalc, N_code))
cov_tree_stim1 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim1 = np.zeros((Ncalc, N_code, N_code))

r_tree_stim2 = np.zeros((Ncalc, N_code))
r_1loop_stim2 = np.zeros((Ncalc, N_code))
cov_tree_stim2 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim2 = np.zeros((Ncalc, N_code, N_code))

r_sim_stim1 = np.zeros((Ncalc, N_code, R))
r_sim_stim2 = np.zeros((Ncalc, N_code, R))
log_odds_tree = np.zeros((Ncalc, R))
log_odds_1loop = np.zeros((Ncalc, R))
log_odds_lin_noise = np.zeros((Ncalc, R))
log_odds_quad_noise = np.zeros((Ncalc, R))

for nn in range(Ncalc):

    print 'progress %: ', float(nn)/float(Ncalc)*100

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale[nn]
    r_tree_stim1[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1
    r_1loop = rates_1loop(W)
    r_1loop_stim1[nn, :] = r_tree_stim1[nn, :] + r_1loop[2*N_ff:2*N_ff+N_code].real*1

    cov_tree_stim1[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1
    cov_1loop, w = two_point_function_fourier_1loop(W)
    cov_1loop_stim1[nn, :, :] = cov_tree_stim1[nn, :, :] + np.real(cov_1loop[2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1

    for r in range(R):
        spktimes, g_vec2 = sim_poisson.sim_poisson(W, tstop, trans, dt)
        spktimes[:, 0] -= trans
        r_temp = np.zeros(N)
        for n in range(N):
            r_temp[n] = sum(spktimes[:, 1] == n) / float(tstop - trans)

        r_sim_stim1[nn, :, r] = r_temp[2*N_ff : 2*N_ff + N_code] * 1

''' calculate theory for stim 2 and mutual inf '''
b_set = raw_input("Enter 'True' after setting b for stim 2 (nonlinear) in params.py")
reload(params)

for nn in range(Ncalc):

    print 'progress %: ', float(nn)/float(Ncalc)*100

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale[nn]
    r_tree_stim2[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1
    r_1loop = rates_1loop(W)
    r_1loop_stim2[nn, :] = r_tree_stim2[nn, :] + r_1loop[2*N_ff:2*N_ff+N_code].real*1

    cov_tree_stim2[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1
    cov_1loop, w = two_point_function_fourier_1loop(W)
    cov_1loop_stim2[nn, :, :] = cov_tree_stim2[nn, :, :] + np.real(cov_1loop[2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1

    for r in range(R):
        ''' log odds of stim 1 vs stim 2 given E population activity '''
        ind = np.where(r_tree_stim1[nn, :] == 0.)[0]
        ind = list(ind)

        r_sim_stim1_ma = np.delete(r_sim_stim1[nn, :, r], ind, axis=0)
        r_tree_stim1_ma = np.delete(r_tree_stim1[nn, :], ind, axis=0)
        r_1loop_stim1_ma = np.delete(r_1loop_stim1[nn, :], ind, axis=0)
        cov_tree_stim1_ma = np.delete(cov_tree_stim1[nn, :, :], ind, axis=0)
        cov_tree_stim1_ma = np.delete(cov_tree_stim1_ma, ind, axis=1)
        cov_1loop_stim1_ma = np.delete(cov_1loop_stim1[nn, :, :], ind, axis=0)
        cov_1loop_stim1_ma = np.delete(cov_1loop_stim1_ma, ind, axis=1)

        p_r1_stim1_tree = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim1_ma, np.diag(np.diag(cov_tree_stim1_ma)))
        p_r1_stim1_1loop = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim1_ma, cov_tree_stim1_ma)
        p_r1_stim1_lin_noise = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim1_ma, cov_tree_stim1_ma)
        p_r1_stim1_quad_noise = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim1_ma, cov_1loop_stim1_ma)

        ind = np.where(r_tree_stim2[nn, :] == 0.)[0]
        ind = list(ind)

        r_sim_stim1_ma = np.delete(r_sim_stim1[nn, :, r], ind, axis=0)
        r_tree_stim2_ma = np.delete(r_tree_stim2[nn, :], ind, axis=0)
        r_1loop_stim2_ma = np.delete(r_1loop_stim2[nn, :], ind, axis=0)
        cov_tree_stim2_ma = np.delete(cov_tree_stim2[nn, :, :], ind, axis=0)
        cov_tree_stim2_ma = np.delete(cov_tree_stim2_ma, ind, axis=1)
        cov_1loop_stim2_ma = np.delete(cov_1loop_stim2[nn, :, :], ind, axis=0)
        cov_1loop_stim2_ma = np.delete(cov_1loop_stim2_ma, ind, axis=1)

        p_r1_stim2_tree = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim2_ma, np.diag(np.diag(cov_tree_stim2_ma)))
        p_r1_stim2_1loop = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim2_ma, cov_tree_stim2_ma)
        p_r1_stim2_lin_noise = mv_norm.logpdf(r_sim_stim1_ma, r_tree_stim2_ma, cov_tree_stim2_ma)
        p_r1_stim2_quad_noise = mv_norm.logpdf(r_sim_stim1_ma, r_1loop_stim2_ma, cov_1loop_stim2_ma)

        p_stim1_r1_tree = p_r1_stim1_tree + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim1_r1_lin_noise = p_r1_stim1_lin_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim1_r1_1loop = p_r1_stim1_1loop + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim1_r1_quad_noise = p_r1_stim1_quad_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        p_stim2_r1_tree = p_r1_stim2_tree + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim2_r1_lin_noise = p_r1_stim2_lin_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim2_r1_1loop = p_r1_stim2_1loop + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim2_r1_quad_noise = p_r1_stim2_quad_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        log_odds_tree[nn, r] = p_stim1_r1_tree - p_stim2_r1_tree
        log_odds_1loop[nn, r] = p_stim1_r1_1loop - p_stim2_r1_1loop
        log_odds_lin_noise[nn, r] = p_stim1_r1_lin_noise - p_stim2_r1_lin_noise
        log_odds_quad_noise[nn, r] = p_stim1_r1_quad_noise - p_stim2_r1_quad_noise

        ''' log odds of stim 1 vs stim 2 given readout activity '''
        p_r1_stim1_tree = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_1loop = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_lin_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim1[nn, -1], cov_tree_stim1[nn, -1, -1])
        p_r1_stim1_quad_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim1[nn, -1], cov_1loop_stim1[nn, -1, -1])

        p_r1_stim2_tree = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_1loop = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_lin_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_tree_stim2[nn, -1], cov_tree_stim2[nn, -1, -1])
        p_r1_stim2_quad_noise = norm.logpdf(r_sim_stim1[nn, -1, r], r_1loop_stim2[nn, -1], cov_1loop_stim2[nn, -1, -1])

        p_stim1_r1_tree = p_r1_stim1_tree + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim1_r1_lin_noise = p_r1_stim1_lin_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim1_r1_1loop = p_r1_stim1_1loop + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim1_r1_quad_noise = p_r1_stim1_quad_noise + smp.log(p_stim1) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        p_stim2_r1_tree = p_r1_stim2_tree + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_tree) * p_stim1 + smp.exp(p_r1_stim2_tree) * p_stim2)
        p_stim2_r1_lin_noise = p_r1_stim2_lin_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_lin_noise) * p_stim1 + smp.exp(p_r1_stim2_lin_noise) * p_stim2)
        p_stim2_r1_1loop = p_r1_stim2_1loop + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_1loop) * p_stim1 + smp.exp(p_r1_stim2_1loop) * p_stim2)
        p_stim2_r1_quad_noise = p_r1_stim2_quad_noise + smp.log(p_stim2) - smp.log(
            smp.exp(p_r1_stim1_quad_noise) * p_stim1 + smp.exp(p_r1_stim2_quad_noise) * p_stim2)

        log_odds_tree_readout[nn, r] = p_stim1_r1_tree - p_stim2_r1_tree
        log_odds_1loop_readout[nn, r] = p_stim1_r1_1loop - p_stim2_r1_1loop
        log_odds_lin_noise_readout[nn, r] = p_stim1_r1_lin_noise - p_stim2_r1_lin_noise
        log_odds_quad_noise_readout[nn, r] = p_stim1_r1_quad_noise - p_stim2_r1_quad_noise

''' plot nonlinear network '''
syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
# ax4.errorbar(syn_scale*weightEE, log_odds_tree.mean(axis=1), yerr=sem(log_odds_tree, axis=1), label='Tree')
# ax4.errorbar(syn_scale*weightEE, log_odds_1loop.mean(axis=1), yerr=sem(log_odds_1loop, axis=1), label='1 Loop')
ax4.errorbar(syn_scale*weightEE, log_odds_lin_noise.mean(axis=1), yerr=sem(log_odds_lin_noise, axis=1), label='Linear Noise')
ax4.errorbar(syn_scale*weightEE, log_odds_quad_noise.mean(axis=1), yerr=sem(log_odds_quad_noise, axis=1), label='Quadratic Noise')

# ax8.errorbar(syn_scale*weightEE, log_odds_tree_readout.mean(axis=1), yerr=sem(log_odds_tree_readout, axis=1), label='Tree')
# ax8.errorbar(syn_scale*weightEE, log_odds_1loop_readout.mean(axis=1), yerr=sem(log_odds_1loop_readout, axis=1), label='1 Loop')
ax8.errorbar(syn_scale*weightEE, log_odds_lin_noise_readout.mean(axis=1), yerr=sem(log_odds_lin_noise_readout, axis=1), label='Linear Noise')
ax8.errorbar(syn_scale*weightEE, log_odds_quad_noise_readout.mean(axis=1), yerr=sem(log_odds_quad_noise_readout, axis=1), label='Quadratic Noise')

''' plot readout example '''
W = W0 * 1.
if Ne > 0:
    W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

if Ni > 0:
    W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

W *= syn_scale[Ncalc_readout_plot]
b_set = raw_input("Enter 'True' after setting b for stim 1 (nonlinear) in params.py")

r_1loop = rates_1loop(W)
cov_1loop, w = two_point_function_fourier_1loop(W)
r_readout_tree_stim1 = rates_ss(W)[Ne] * 1
r_readout_1loop_stim1 = r_readout_tree_stim1 + r_1loop[Ne].real * 1
sig_readout_tree_stim1 = np.real(two_point_function_fourier(W)[0])[Ne, Ne]*1
sig_readout_1loop_stim1 = sig_readout_tree_stim1 + cov_1loop[Ne, Ne].real*1

r_plot = np.arange(0, 1000, .1) / 1000.
ax12.plot(r_plot, norm.pdf(r_plot, r_readout_tree_stim1, sig_readout_tree_stim1), 'k')
ax12.plot(r_plot, norm.pdf(r_plot, r_readout_1loop_stim1.real, sig_readout_1loop_stim1.real), 'r')

b_set = raw_input("Enter 'True' after setting b for stim 2 (nonlinear) in params.py")

r_1loop = rates_1loop(W)
cov_1loop, w = two_point_function_fourier_1loop(W)
r_readout_tree_stim2 = rates_ss(W)[Ne] * 1
r_readout_1loop_stim2 = r_readout_tree_stim2 + r_1loop[Ne] * 1
sig_readout_tree_stim2 = np.real(two_point_function_fourier(W)[0])[Ne, Ne]*1
sig_readout_1loop_stim2 = sig_readout_tree_stim2 + cov_1loop[Ne, Ne].real*1

ax12.plot(r_plot, norm.pdf(r_plot, r_readout_tree_stim2, sig_readout_tree_stim2), 'k--')
ax12.plot(r_plot, norm.pdf(r_plot, r_readout_1loop_stim2.real, sig_readout_1loop_stim2.real), 'r--')


'''
format plots
'''
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

ax4.set_ylabel('Log odds of stim 1 vs stim 2 | E pop activity')
ax8.set_ylabel('Log odds of stim 1 vs stim 2 | readout activity')

if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/structure_driven_activity/'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/'

savefile = os.path.join(save_dir, 'Fig_decoding_pop.eps')
fig1.savefig(savefile)
fig1.show()

savefile = os.path.join(save_dir, 'Fig_decoding_readout.eps')
fig2.savefig(savefile)
fig2.show()


savefile = os.path.join(save_dir, 'Fig_decoding_readoout_prob.eps')
fig3.savefig(savefile)
fig3.show()