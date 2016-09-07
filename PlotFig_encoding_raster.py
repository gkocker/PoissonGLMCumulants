'''
Plot figure: rasters of responses to stim 1 and stim 2
params for stim 1: b[N_ff:2*N_ff] = 0 (2nd input assembly off)
params for sti 2: b[:N_ff] = 0 (1st input assembly off)
'''


''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj

from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier
from theory import two_point_function_fourier_1loop
from theory import two_point_function_fourier_pop
from theory import two_point_function_fourier_pop_1loop

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import norm


import sim_poisson

import time
import os
import sys


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/encoding_assembly/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

plot_linear = raw_input("Enter 'True' to plot linear, 'False' to plot nonlinear")

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
N_ff = par.N_ff

''' set up subplot grid '''
fig = plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=.95, bottom=.65, hspace=0.1, left=0.05, right=0.95)
ax1 = plt.subplot(gs1[0, 0])
ax2 = plt.subplot(gs1[0, 1])
ax3 = plt.subplot(gs1[0, 2])

gs2 = gridspec.GridSpec(2, 1)
gs2.update(top=.6, bottom=.05, hspace=0.1, left=0.05, right=0.95)
ax4 = plt.subplot(gs2[0, 0])
ax5 = plt.subplot(gs2[1, 0])


# '''
# plot transfer functions
# '''
# x = np.arange(-1, 1, .01)
# if plot_linear == 'True':
#     y = gain*x
# else:
#     y = gain * x**2
#
# y[np.where(x<0)[0]] = 0
# ax1.plot(x, y, 'k', linewidth=2)
# ax1.set_xlabel('Membrane voltage', fontsize=12)
# ax1.set_ylabel('Firing rate', fontsize=12)
# ax1.set_xlim((-1, 1))
# ax1.set_ylim((-.001, .1))
# ax1.set_xticks([0])
# ax1.set_yticks([])

# y[np.where(x<0)[0]] = 0
# ax2.plot(x, y, 'k', linewidth=2)
# ax2.set_xlabel('Membrane voltage', fontsize=12)
# ax2.set_ylabel('Firing rate', fontsize=12)
# ax2.set_xlim((-1, 1))
# ax2.set_ylim((-.001, .1))
# ax2.set_xticks([0])
# ax2.set_yticks([])


''' load or generate adjacency matrix '''
W0_path = os.path.join(save_dir, 'W0_2layer.npy')

if os.path.exists(W0_path):
    W0 = np.load(os.path.join(save_dir, 'W0_2layer.npy'))
else:
    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)

    if Ne > 0 and N_ff > 0: # make first 2 groups of 50 neurons a projection layer
        W0_ff = np.random.rand(Ne-2*N_ff, N_ff)
        W0_ff2 = W0_ff.copy()
        W0_ff[W0_ff > .5] = 0.
        W0_ff[W0_ff != 0] = 1.

        W0_ff2[W0_ff2 > .25] = 0.
        W0_ff2[W0_ff2 != 0] = 1.

        W0[:Ne, :2*N_ff] = 0
        W0[ind_pop, :N_ff] += W0_ff

        W0[ind_pop, N_ff:2*N_ff] += W0_ff2

        W0[:2*N_ff, :] = 0

    savefile = os.path.join(save_dir, 'W0_2layer.npy')
    np.save(savefile, W0)

'''
plot connectivity
'''
ax1.imshow(W0, cmap='gray_r')
ax1.set_ylim((0, N))
ax1.set_xlabel('Presynaptic Neuron')
ax1.set_ylabel('Postsynaptic Neuron')

''' generate sims for weak stimulus and linear network '''
if plot_linear == 'True':
    syn_scale = [4.]  # for linear
else:
    # syn_scale = [28.]  # for quadratic
    syn_scale = [20.]

dt = .01 * tau
trans = 50. * tau
tstop = 6000. + trans

### generate scaled weight matrix from frozen connectivity realization
W = W0 * 1.
if Ne > 0:
    W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

if Ni > 0:
    W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

W *= syn_scale[0]


print 'running sims 1'
if plot_linear == 'True':
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-linear in phi.py")
else:
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-quadratic in phi.py")

b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")

spktimes, g_vec1 = sim_poisson.sim_poisson(W, tstop, trans, dt)

print 'plotting raster 1'
spktimes[:, 0] -= trans
numspikes = spktimes.shape[0]

for i in range(0, numspikes):
    ax2.plot(spktimes[i, 0]/1000., spktimes[i, 1], 'k.', markersize=.1)

ax3.set_yticks([0, Ne-1])
ax2.set_xticks([0, (tstop-trans)/1000.])
# ax2.set_ylabel('Neuron', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylim((0, N))
ax2.set_xlim((0, (tstop-trans)/1000.))
ax2.set_title('Stimulus A')

if plot_linear == 'True':
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-linear in phi.py")
else:
    phi_set = raw_input("Enter 'True' after checking that transfer is threshold-quadratic in phi.py")

b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")
reload(params)
b = par.b

spktimes, g_vec2 = sim_poisson.sim_poisson(W, tstop, trans, dt)

print 'plotting raster 2'
spktimes[:, 0] -= trans
numspikes = spktimes.shape[0]

for i in range(0, numspikes):
    ax3.plot(spktimes[i, 0]/1000., spktimes[i, 1], 'k.', markersize=.1)

ax3.set_yticks([0, Ne-1])
ax3.set_xticks([0, (tstop-trans)/1000.])
ax3.set_yticklabels([])
# ax4.set_ylabel('Neuron', fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylim((0, N))
ax3.set_xlim((0, (tstop-trans)/1000.))
ax3.set_title('Stimulus B')



''' plot pop mean rate pdfs '''
N_code = (Ne - N_ff)/2
ind_pop = range(N_ff, N_ff+N_code)
b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")

tstop = 60000. + trans
T = 200.
spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
spktimes[:, 0] -= trans

T_start = range(0, int(tstop-trans), int(T/2))
r_sim_stim1 = np.zeros(len(T_start))
for i, t in enumerate(T_start):
    spk_temp = spktimes[spktimes[:, 0] >= t]
    spk_temp = spk_temp[spk_temp[:, 0] < t+T]
    r_temp = np.zeros(N)
    for n in range(N):
        r_temp[n] = sum(spk_temp[:, 1] == n)

    r_sim_stim1[i] = np.mean(r_temp[ind_pop])


# for r in range(R):
#     spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
#     spktimes[:, 0] -= trans
#     r_temp = np.zeros(N)
#     for n in range(N):
#         r_temp[n] = sum(spktimes[:, 1] == n)
#
#     r_sim_stim1[r] = np.sum(r_temp[ind_pop]) / float(N_code)


if plot_linear == 'True':
    hist, bins = np.histogram(r_sim_stim1, bins=np.arange(0, 2, .04), density=True)
else:
    hist, bins = np.histogram(r_sim_stim1, bins=np.arange(0, 2, .02), density=True)

ax4.plot(bins[:-1], hist, 'k', linewidth=1)
ax5.plot(bins[:-1], hist, 'k', linewidth=1)

r_pop_tree_stim1 = np.sum(rates_ss(W)[ind_pop]) / float(N_code)*(T)
r_pop_1loop_stim1 = r_pop_tree_stim1 + np.sum(rates_1loop(W)[ind_pop].real) / float(N_code) *(T)
sig_pop_tree_stim1 = np.sqrt(two_point_function_fourier_pop(W, ind_pop).real * (T))
sig_pop_1loop_stim1 = np.sqrt((two_point_function_fourier_pop(W, ind_pop).real + two_point_function_fourier_pop_1loop(W, ind_pop).real) * (T))

if plot_linear == 'True':
    r_plot = np.arange(0, 5, .005)
else:
    r_plot = np.arange(0, 1, .001)

if plot_linear == 'True':
    ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim1, sig_pop_tree_stim1), 'k', linewidth=2, label='Stim. A')
else:
    # ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim1, sig_pop_tree_stim1), 'k', linewidth=2)
    ax4.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim1, sig_pop_1loop_stim1), 'r', linewidth=2)

sig_pop_tree_stim1 = r_pop_tree_stim1 / float(N_code)
sig_pop_1loop_stim1 = r_pop_1loop_stim1 / float(N_code)
if plot_linear == 'True':
    ax5.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim1, sig_pop_tree_stim1), 'k', linewidth=2, label='Stim. A')
else:
    # ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim1, sig_pop_tree_stim1), 'k', linewidth=2)
    ax5.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim1, sig_pop_1loop_stim1), 'r', linewidth=2)


b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")

spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
spktimes[:, 0] -= trans

T_start = range(0, int(tstop-trans), int(T/2))
r_sim_stim2 = np.zeros(len(T_start))
for i, t in enumerate(T_start):
    spk_temp = spktimes[spktimes[:, 0] >= t]
    spk_temp = spk_temp[spk_temp[:, 0] < t+T]
    r_temp = np.zeros(N)
    for n in range(N):
        r_temp[n] = sum(spk_temp[:, 1] == n)

    r_sim_stim2[i] = np.mean(r_temp[ind_pop])

if plot_linear == 'True':
    hist, bins = np.histogram(r_sim_stim2, bins=np.arange(0, 2, .04), density=True)
else:
    hist, bins = np.histogram(r_sim_stim2, bins=np.arange(0, 2, .02), density=True)

ax4.plot(bins[:-1], hist, 'b', linewidth=1)
ax5.plot(bins[:-1], hist, 'b', linewidth=1)

r_pop_tree_stim2 = np.sum(rates_ss(W)[ind_pop]) / float(N_code) * (T)
r_pop_1loop_stim2 = r_pop_tree_stim2 + np.sum(rates_1loop(W)[ind_pop].real) / float(N_code)*(T)
sig_pop_tree_stim2 = np.sqrt(two_point_function_fourier_pop(W, ind_pop).real * (T))
sig_pop_1loop_stim2 = np.sqrt( (two_point_function_fourier_pop(W, ind_pop).real + two_point_function_fourier_pop_1loop(W, ind_pop).real) * (T))


if plot_linear == 'True':
    ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim2, sig_pop_tree_stim2), 'b', linewidth=2, label='Stim. B')
    ax4.legend(loc=0)
else:
    # ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim2, sig_pop_tree_stim2), 'k', linewidth=2)
    ax4.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim2.real, sig_pop_1loop_stim2), 'b', linewidth=2)

ax4.set_yticklabels([])
ax4.set_ylabel('Probability')
ax4.set_xlabel('Spike Count. T=200 ms')

sig_pop_tree_stim2 = r_pop_tree_stim2 / float(N_code)
sig_pop_1loop_stim2 = r_pop_1loop_stim2 / float(N_code)
if plot_linear == 'True':
    ax5.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim2, sig_pop_tree_stim2), 'b', linewidth=2, label='Stim. A')
else:
    # ax4.plot(r_plot, norm.pdf(r_plot, r_pop_tree_stim2, sig_pop_tree_stim2), 'k', linewidth=2)
    ax5.plot(r_plot, norm.pdf(r_plot, r_pop_1loop_stim2, sig_pop_1loop_stim2), 'b', linewidth=2)

ax4.spines['bottom'].set_visible(False)
ax4.set_xticks([])

for ax in [ax1, ax2, ax3, ax4, ax5]:

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/encoding_assembly'

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_linear.eps')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_nonlinear.eps')


plt.savefig(savefile)
plt.show()

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_linear.png')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_nonlinear.png')


plt.savefig(savefile)
plt.show()