'''
Plot figure: coding metrics between activity and stimulus for different versions of the theory

'''

''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from phi import phi_prime

from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier_pop
from theory import two_point_function_fourier_pop_1loop
import os
import sys


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gocker/Documents/projects/field_theory_spiking/encoding_assembly/'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/encoding_assembly/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


plot_linear = raw_input("Enter 'True' for threshold-linear transfer function or 'False' to plot nonlinear")
if plot_linear == 'True':
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-linear in phi.py")
else:
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-quadratic in phi.py")

# load_sims_input = raw_input("Enter 'True' to try to load sims, False to run sims")
# run_sims = False
# if load_sims_input == 'True':
#     if plot_linear == 'True':
#         r_sim_stim2 = np.load(os.path.join(save_dir, 'popsum/r_sim_stim2_linear.npy'))
#     elif plot_linear == 'False':
#         r_sim_stim2 = np.load(os.path.join(save_dir, 'popsum/r_sim_stim2_nonlinear.npy'))
#     else:
#         print 'no sims to load, running sims'
#         run_sims = True
# elif load_sims_input == 'False':
#     run_sims = True
#
# if run_sims:
#     r_sim_stim2 = np.zeros((Ncalc, R, N))


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

        ''' two subsets of excitatory receive different inputs'''
        W0_assembly1 = np.random.rand((Ne-N_ff)/2, N_ff/2)
        W0_assembly2 = np.random.rand((Ne-N_ff)/2, N_ff/2)
        W0_assembly1[W0_assembly1 > .5] = 0.
        W0_assembly1[W0_assembly1 != 0.] = 1.
        W0_assembly2[W0_assembly2 > .5] = 0.
        W0_assembly2[W0_assembly2 != 0] = 1.

        W0[N_ff:, :N_ff] = 0.
        W0[N_ff:N_ff+int(np.floor((Ne-N_ff)/2.)), :N_ff/2] += W0_assembly1
        W0[N_ff+int(np.floor((Ne-N_ff)/2.)):Ne, N_ff/2:N_ff] += W0_assembly2

        W0[:N_ff, :] = 0.

        ''' assemblies in the recurrent/coding network'''
        # p_in = pEE + 3.*pEE/4.
        # p_cross = pEE/4.
        # W0EE = np.random.rand((Ne-N_ff), (Ne-N_ff))
        # W0EE[W0EE > p_cross] = 0.
        # W0EE[W0EE != 0] = 1.
        # W0[N_ff:Ne, N_ff:Ne] = W0EE
        #
        # W0_assembly1 = np.random.rand((Ne-N_ff)/2, (Ne-N_ff)/2)
        # W0_assembly2 = np.random.rand((Ne-N_ff)/2, (Ne-N_ff)/2)
        # W0_assembly1[W0_assembly1 > p_in] = 0.
        # W0_assembly1[W0_assembly1 != 0.] = 1.
        # W0_assembly2[W0_assembly2 > p_in]  = 0.
        # W0_assembly2[W0_assembly2 != 0.] = 1.
        #
        # W0[N_ff:N_ff+(Ne-N_ff)/2, N_ff:N_ff+(Ne-N_ff)/2] = W0_assembly1
        # W0[N_ff+(Ne-N_ff)/2:Ne, N_ff+(Ne-N_ff)/2:Ne] = W0_assembly2
        # W0 -= np.diag(np.diag(W0))


    savefile = os.path.join(save_dir, 'W0_2layer.npy')
    np.save(savefile, W0)


''' simulation parameters '''
R = 20000
T = 200
trans = 200
tstop = R * T/2 + trans
T_start = np.arange(0, tstop-trans-T/2, T/2)

dt = .02*tau

''' plot log odds of stim given activity as synaptic strength scales '''
Ncalc = 40
N_code = (Ne - N_ff)/2
ind_pop = np.arange(N_ff, N_ff+N_code)

p_stim1 = 0.5
p_stim2 = 0.5

print 'ranging over synaptic weights for nonlinear network, stim 1'
if plot_linear == 'True':
    syn_scale = np.linspace(0., 12., Ncalc)  # for linear
else:
    syn_scale = np.linspace(0., 36., Ncalc)  # for quadratic

r_tree_stim1 = np.zeros((Ncalc))
r_1loop_stim1 = np.zeros((Ncalc))
cov_tree_stim1 = np.zeros((Ncalc))
cov_1loop_stim1 = np.zeros((Ncalc))

r_tree_stim2 = np.zeros((Ncalc))
r_1loop_stim2 = np.zeros((Ncalc))
cov_tree_stim2 = np.zeros((Ncalc))
cov_1loop_stim2 = np.zeros((Ncalc))

log_odds_lin_noise = np.zeros((Ncalc, R))
log_odds_quad_noise = np.zeros((Ncalc, R))
log_odds_tree = np.zeros((Ncalc, R))
log_odds_1loop = np.zeros((Ncalc, R))

condent_lin_noise = np.zeros((Ncalc, 2))
condent_quad_noise = np.zeros((Ncalc, 2))

KL_stim_lin_noise = np.zeros((Ncalc))
KL_stim_quad_noise = np.zeros((Ncalc))
KL_stim_tree = np.zeros((Ncalc))
KL_stim_1loop = np.zeros((Ncalc))

percent_correct_lin_noise = np.zeros((Ncalc))
percent_correct_quad_noise = np.zeros((Ncalc))
percent_correct_tree = np.zeros((Ncalc))
percent_correct_1loop = np.zeros((Ncalc))

b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")
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
    r_tree_stim1[nn] = np.mean(rates_ss(W)[ind_pop])*T
    r_1loop = rates_1loop(W)
    r_1loop_stim1[nn] = r_tree_stim1[nn] + np.mean(r_1loop[ind_pop]).real*T

    cov_tree_stim1[nn] = two_point_function_fourier_pop(W, ind_pop)[0].real*T
    cov_1loop_stim1[nn] = cov_tree_stim1[nn] + two_point_function_fourier_pop_1loop(W, ind_pop).real*T


''' calculate theory for stim 2 and mutual inf '''
b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")
reload(params)
Nplot = Ncalc-1

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

    r_tree_stim2[nn] = np.mean(rates_ss(W)[ind_pop])*T
    r_1loop_stim2[nn] = r_tree_stim2[nn] + np.mean(rates_1loop(W)[ind_pop].real)*T

    cov_tree_stim2[nn] = two_point_function_fourier_pop(W, ind_pop).real*T
    cov_1loop_stim2[nn] = cov_tree_stim2[nn] + two_point_function_fourier_pop_1loop(W, ind_pop).real*T

    g = np.dot(W, rates_ss(W)) + b
    if not np.isnan(g).any():
        stab_mat_mft2 = np.dot(np.diag(phi_prime(g, gain)), W)
        ub, v2 = np.linalg.eig(stab_mat_mft2)
    else:
        ub = 2*np.ones(N)

    if np.amax(np.abs(ub)) < 1.:

        # for r in range(R):
        #     spktimes, g_vec2 = sim_poisson(W, tstop, trans, dt)
        #     spktimes[:, 0] -= trans
        #     r_temp = np.zeros(N)
        #     for n in range(N):
        #         r_temp[n] = sum(spktimes[:, 1] == n)
        #
        #     r_sim_stim2[nn, r] = np.mean(r_temp[ind_pop])

        sig1 = r_tree_stim1[nn] / float(N_code)
        sig2 = r_tree_stim2[nn] / float(N_code)

        KL_stim_tree[nn] = np.log(np.sqrt(sig1)/np.sqrt(sig2)) + (sig2+(r_tree_stim2[nn]-r_tree_stim1[nn])**2) / (2*sig1) - 1./2.

        sig1 = cov_tree_stim1[nn]
        sig2 = cov_tree_stim2[nn]

        KL_stim_lin_noise[nn] = np.log(np.sqrt(sig1)/np.sqrt(sig2)) + (sig2+(r_tree_stim2[nn]-r_tree_stim1[nn])**2) / (2*sig1) - 1./2.
        KL_stim_1loop[nn] = np.log(np.sqrt(sig1)/np.sqrt(sig2)) + (sig2+(r_1loop_stim2[nn]-r_1loop_stim1[nn])**2) / (2*sig1) - 1./2.

        sig1 = cov_1loop_stim1[nn]
        sig2 = cov_1loop_stim2[nn]

        KL_stim_quad_noise[nn] = np.log(np.sqrt(sig1)/np.sqrt(sig2)) + (sig2+(r_1loop_stim2[nn]-r_1loop_stim1[nn])**2) / (2*sig1) - 1./2.

    else:
        Nplot = nn
        break

# from sim_poisson import sim_poisson
# for nn in range(0, Ncalc, 2):
#
#     print 'progress %: ', float(nn)/float(Ncalc)*100
#
#     ### generate scaled weight matrix from frozen connectivity realization
#     W = W0 * 1.
#     if Ne > 0:
#         W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
#         W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]
#
#     if Ni > 0:
#         W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
#         W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]
#
#     W *= syn_scale[nn]
#
#     g = np.dot(W, rates_ss(W)) + b
#     if not np.isnan(g).any():
#         stab_mat_mft2 = np.dot(np.diag(phi_prime(g, gain)), W)
#         ub, v2 = np.linalg.eig(stab_mat_mft2)
#     else:
#         ub = 2 * np.ones(N)
#
#     if np.amax(np.abs(ub)) < 1.:
#
#         ''' calculate discriminants for percent correct'''
#         if run_sims:
#             spktimes, g_vec2 = sim_poisson(W, tstop, trans, dt)
#             spktimes[:, 0] -= trans
#             #
#             for r, t in enumerate(T_start):
#                 ind_start = np.where(spktimes[:, 0] >= t)[0][0]
#                 ind_end = np.where(spktimes[:, 0] < t + T)[0][-1]
#                 for n in range(N):
#                     r_sim_stim2[nn, r, n] = sum(spktimes[ind_start:ind_end, 1] == n)
#
#                 x = np.mean(r_sim_stim2[nn, r, ind_pop])
#
#                 mu1 = r_tree_stim1[nn]
#                 mu2 = r_tree_stim2[nn]
#
#                 sig1 = r_tree_stim1[nn] / float(N_code)
#                 sig2 = r_tree_stim2[nn] / float(N_code)
#                 log_odds_tree[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                 np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 sig1 = cov_tree_stim1[nn]
#                 sig2 = cov_tree_stim2[nn]
#                 log_odds_lin_noise[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                 np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 mu1 = r_1loop_stim1[nn]
#                 mu2 = r_1loop_stim2[nn]
#                 log_odds_1loop[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                 np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 sig1 = cov_1loop_stim1[nn]
#                 sig2 = cov_1loop_stim2[nn]
#                 log_odds_quad_noise[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                 np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#         else:
#             for r in range(R):
#                 x = np.mean(r_sim_stim2[nn, r, ind_pop])
#
#                 mu1 = r_tree_stim1[nn]
#                 mu2 = r_tree_stim2[nn]
#
#                 sig1 = r_tree_stim1[nn] / float(N_code)
#                 sig2 = r_tree_stim2[nn] / float(N_code)
#                 log_odds_tree[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                     np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 sig1 = cov_tree_stim1[nn]
#                 sig2 = cov_tree_stim2[nn]
#                 log_odds_lin_noise[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                     np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 mu1 = r_1loop_stim1[nn]
#                 mu2 = r_1loop_stim2[nn]
#                 log_odds_1loop[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                     np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#
#                 sig1 = cov_1loop_stim1[nn]
#                 sig2 = cov_1loop_stim2[nn]
#                 log_odds_quad_noise[nn, r] = np.log(p_stim2) + norm.pdf(x, loc=mu2, scale=sig2) - (
#                     np.log(p_stim1) + norm.pdf(x, loc=mu1, scale=sig1))
#


''' set up subplot grid for log odds'''
fig1 = plt.figure(figsize=(6, 2))
gs1 = gridspec.GridSpec(1, 1)
gs1.update(top=.85, bottom=.05, hspace=0.1, left=0.05, right=0.25)
ax1 = plt.subplot(gs1[0, 0])

gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax2 = plt.subplot(gs2[0, 0])
ax3 = plt.subplot(gs2[0, 1])

'''
plot transfer functions
'''
x = np.arange(-1, 2, .01)
if plot_linear == 'True':
    y = gain*x
else:
    y = gain * x**2

y[np.where(x<0)[0]] = 0
for ax in [ax1]:

    ax.plot(x, y, 'k', linewidth=2, label='Excitatory')
    ax.set_xlabel('Membrane voltage', fontsize=12)
    ax.set_ylabel('Firing rate', fontsize=12)
    ax.set_xlim((-1, 2))
    ax.set_ylim((-.001, .5))
    ax.set_xticks([0])
    ax.set_yticks([])

''' plot log odds and D_KL vs synaptic weight'''

# ax3.errorbar(syn_scale[:Nplot:2]*weightEE, log_odds_lin_noise[:Nplot:2].mean(axis=1), yerr=np.std(log_odds_lin_noise[:Nplot:2], axis=1) / np.sqrt(R), fmt=None, color='k', elinewidth=2, ecolor='k')
# ax3.errorbar(syn_scale[:Nplot:2]*weightEE, log_odds_tree[:Nplot:2].mean(axis=1), yerr=np.std(log_odds_tree[:Nplot:2], axis=1) / np.sqrt(R), fmt=None, color='k', elinewidth=2, ecolor='k')
# ax3.plot(syn_scale[:Nplot:2]*weightEE, log_odds_lin_noise[:Nplot:2].mean(axis=1), label='Linear Noise', color='k', linewidth=2)
# ax3.plot(syn_scale[:Nplot:2]*weightEE, log_odds_tree[:Nplot:2].mean(axis=1), 'k--', label='Tree', linewidth=2)
# ax3.set_ylabel('Log odds of B vs A')
# ax3.set_xlim((0, syn_scale[Nplot]*weightEE))

# ax3.plot(syn_scale[:Nplot:2]*weightEE, np.sum(log_odds_lin_noise[:Nplot:2, :] > 0, axis=1) / float(R), label='Linear Noise', color='k', linewidth=2)
# ax3.plot(syn_scale[:Nplot:2]*weightEE, np.sum(log_odds_tree[:Nplot:2, :] > 0, axis=1) / float(R), 'k--', label='Tree', linewidth=2)
# ax3.set_ylabel('Percent correct classification')



# ax2.plot(syn_scale[:Nplot]*weightEE, KL_stim_lin_noise[:Nplot], 'k', linewidth=2)
# ax2.plot(syn_scale[:Nplot]*weightEE, KL_stim_tree[:Nplot], 'k--', linewidth=2)

ax2.plot(syn_scale[:Nplot]*weightEE, r_tree_stim1[:Nplot] - r_tree_stim2[:Nplot], 'k', linewidth=2)
ax3.plot(syn_scale[:Nplot]*weightEE, cov_tree_stim1[:Nplot]/cov_tree_stim2[:Nplot], 'k', linewidth=2)
ax3.plot(syn_scale[:Nplot]*weightEE, r_tree_stim1[:Nplot]/r_tree_stim2[:Nplot], 'k--', linewidth=2)

if plot_linear != 'True':
    # # ax3.errorbar(syn_scale[:Nplot:2]*weightEE, log_odds_quad_noise[:Nplot:2].mean(axis=1), yerr=np.std(log_odds_quad_noise[:Nplot:2], axis=1) / np.sqrt(R), fmt=None, color='r', elinewidth=2, ecolor='r')
    # ax3.errorbar(syn_scale[:Nplot:2]*weightEE, log_odds_1loop[:Nplot:2].mean(axis=1), yerr=np.std(log_odds_1loop[:Nplot:2], axis=1) / np.sqrt(R), fmt=None, color='r', elinewidth=2, ecolor='r')
    # # ax3.plot(syn_scale[:Nplot:2]*weightEE, log_odds_quad_noise[:Nplot:2].mean(axis=1), label='Quadratic Noise', color='r', linewidth=2)
    # ax3.plot(syn_scale[:Nplot:2]*weightEE, log_odds_1loop[:Nplot:2].mean(axis=1), 'r', label='1 loop', linewidth=2)

    # ax3.plot(syn_scale[:Nplot:2]*weightEE, np.sum(log_odds_quad_noise[:Nplot:2, :] > 0, axis=1) / float(R), label='Quadratic Noise',
    #          color='r', linewidth=2)
    # ax3.plot(syn_scale[:Nplot:2]*weightEE, np.sum(log_odds_1loop[:Nplot:2, :] > 0, axis=1) / float(R), 'r', label='1 loop', linewidth=2)

    # # ax2.plot(syn_scale[:Nplot] * weightEE, KL_stim_quad_noise[:Nplot], 'r', linewidth=2)
    # ax2.plot(syn_scale[:Nplot] * weightEE, KL_stim_1loop[:Nplot], 'r', linewidth=2)

    ax2.plot(syn_scale[:Nplot]*weightEE, r_1loop_stim1[:Nplot] - r_1loop_stim2[:Nplot], 'r', linewidth=2)
    # ax3.plot(syn_scale[:Nplot]*weightEE, )

ax2.set_xlabel('Exc.-exc. synaptic weight (mV)')
# ax2.set_ylabel(r'$D_{KL}(p(r|B,) || p(r|A))$')
ax2.set_ylabel('r | B - r|A')
ax3.set_ylabel(r'$\sigma_B^2 / \sigma_A^2$')
ax2.set_xlim((0, syn_scale[Nplot]*weightEE))
ax3.set_xlim((0, syn_scale[Nplot]*weightEE))

ax2.legend(loc=0)


# ''' plot pop mean rate pdfs '''
# W = W0 * 1.
# if Ne > 0:
#     W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
#     W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]
#
# if Ni > 0:
#     W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
#     W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]
#
# Ncalc_readout_plot = int(np.floor(Ncalc/3.))
# W *= syn_scale[Ncalc_readout_plot]
#
# b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")
#
# R = 1
# r_sim_stim1 = np.zeros(R)
# for r in range(R):
#     spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
#     spktimes[:, 0] -= trans
#     r_temp = np.zeros(N)
#     for n in range(N):
#         r_temp[n] = sum(spktimes[:, 1] == n) / float(tstop - trans)
#
#     r_sim_stim1[r] = np.sum(r_temp[2*N_ff:Ne]) / float(N_code)
#
#
# r_pop_tree_stim1 = np.sum(rates_ss(W)[2*N_ff:Ne]) / float(N_code)
# r_pop_1loop_stim1 = r_pop_tree_stim1 + np.sum(rates_1loop(W)[2*N_ff:Ne].real) / float(N_code)
# sig_pop_tree_stim1 = two_point_function_fourier_pop(W, range(2*N_ff, Ne)).real
# sig_pop_1loop_stim1 = sig_pop_tree_stim1 + two_point_function_fourier_pop_1loop(W, range(2*N_ff, Ne)).real
#
# r_plot = np.arange(0, 50, .001) / 1000.
# ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim1, sig_pop_tree_stim1), 'k', linewidth=2)
# ax4.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim1, sig_pop_1loop_stim1), 'r', linewidth=2)
#
# b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")
#
# r_sim_stim2 = np.zeros(R)
# for r in range(R):
#     spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
#     spktimes[:, 0] -= trans
#     r_temp = np.zeros(N)
#     for n in range(N):
#         r_temp[n] = sum(spktimes[:, 1] == n) / float(tstop - trans)
#
#     r_sim_stim2[r] = np.sum(r_temp[2*N_ff:Ne]) / float(N_code)
#
# r_pop_tree_stim2 = np.sum(rates_ss(W)[2*N_ff:Ne]) / float(N_code)
# r_pop_1loop_stim2 = r_pop_tree_stim2 + np.sum(rates_1loop(W)[2*N_ff:Ne].real) / float(N_code)
# sig_pop_tree_stim2 = two_point_function_fourier_pop(W, range(2*N_ff, Ne)).real
# sig_pop_1loop_stim2 = sig_pop_tree_stim2 + two_point_function_fourier_pop_1loop(W, range(2*N_ff, Ne)).real
#
#
# ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim2, sig_pop_tree_stim2), 'k--', linewidth=2)
# ax4.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim2.real, sig_pop_1loop_stim2), 'r--', linewidth=2)
#

'''
format plots
'''
for ax in [ax1, ax2, ax3]:

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

# ax4.set_ylabel('Log odds of stim 1 vs stim 2 | E pop activity')


# fig3 = plt.figure()
# plt.plot(syn_scale[:Nplot]*weightEE, KL_stim_lin_noise[:Nplot], 'k', syn_scale[:Nplot]*weightEE, KL_stim_lin_noise[:Nplot], 'r', linewidth=2)
# plt.xlabel('Exc.-exc. synaptic weight (mV)')
# plt.ylabel(r'$D_{KL}(p(r|A,), p(r|B))$')
# if plot_linear == 'True':
#     plt.title('Linear')
# else:
#     plt.title('Nonlinear')

# fig4 = plt.figure()
# plt.plot(syn_scale[:Nplot]*weightEE, condent_lin_noise[:Nplot, 0] - condent_lin_noise[:Nplot, 1], 'k', linewidth=2)
# plt.plot(syn_scale[:Nplot]*weightEE, condent_quad_noise[:Nplot, 0] - condent_quad_noise[:Nplot, 1], 'r', linewidth=2)
# plt.xlabel('Exc.-exc. synaptic weight (mV)')
# plt.ylabel(r'$H(p(r|A)) - H(p(r|B))$')
# if plot_linear == 'True':
#     plt.title('Linear')
# else:
#     plt.title('Nonlinear')


''' save plots'''

if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly/popsum'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/encoding_assembly/popsum'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# savefile = os.path.join(save_dir, 'Fig_decoding_pop_nonlinear.eps')
# fig1.savefig(savefile)
# fig1.show()

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'r_sim_stim2_linear.npy')
    np.save(savefile, r_sim_stim2)
elif plot_linear == 'False':
    savefile = os.path.join(save_dir, 'r_sim_stim2_nonlinear.npy')
    np.save(savefile, r_sim_stim2)
else:
    print 'check transfer function and save rates'


if plot_linear == 'True' and Ni > 0:
    savefile = os.path.join(save_dir, 'Fig_decoding_popsum_linear_Ncode='+str(len(ind_pop))+'.eps')
elif plot_linear != 'True' and Ni > 0:
    savefile = os.path.join(save_dir, 'Fig_decoding_popsum_nonlinear_Ncode='+str(len(ind_pop))+'.eps')
elif plot_linear == 'True' and Ni == 0:
    savefile = os.path.join(save_dir, 'Fig_decoding_Ni=0_linear.eps')
elif plot_linear != 'True' and Ni == 0:
    savefile = os.path.join(save_dir, 'Fig_decoding_Ni=0_nonlinear.eps')

fig1.savefig(savefile)
fig1.show()

# if plot_linear == 'True':
#     savefile = os.path.join(save_dir, 'Fig_coding_pop_dist'
#                                       '_linear.eps')
# else:
#     savefile = os.path.join(save_dir, 'Fig_coding_pop_dist_nonlinear.eps')
# fig2.savefig(savefile)
# fig2.show()