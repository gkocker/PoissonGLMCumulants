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
import raster

import time
import os
import sys


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/encoding_assembly/'
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
        W0_ff = np.random.rand(N, N)
        W0_ff[W0_ff > .5] = 0.
        W0_ff[W0_ff != 0] = 1.

        W0[:, :N_ff] = 0
        W0[N_ff:N_ff+N_code, :N_ff/2] += W0_ff[N_ff:N_ff+N_code, :N_ff/2]

        W0[N_ff+N_code:Ne, N_ff/2:N_ff] += W0_ff[N_ff+N_code:Ne, N_ff/2:N_ff]

    savefile = os.path.join(save_dir, 'W0_2layer.npy')
    np.save(savefile, W0)

'''
plot connectivity
'''
ax1.imshow(W0, cmap='gray_r')
ax1.set_ylim((0, N))
ax1.set_xlabel('Presynaptic Neuron')
ax1.set_ylabel('Postsynaptic Neuron')

if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/encoding_assembly'

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'Fig_coding_adjacency.eps')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_adjacency.eps')


plt.savefig(savefile)
plt.show()
plt.close(fig)

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
    savefile = os.path.join(save_dir, 'Fig_coding_rasters1.eps')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_rasters1.eps')

plt.savefig(savefile)
plt.close(fig)

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




if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/encoding_assembly'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/field_theory_spiking/encoding_assembly'

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_linear.eps')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_rasters_nonlinear.eps')


plt.savefig(savefile)
plt.show()

if plot_linear == 'True':
    savefile = os.path.join(save_dir, 'Fig_coding_rasters2.eps')
else:
    savefile = os.path.join(save_dir, 'Fig_coding_rasters2.eps')

plt.savefig(savefile)
plt.close(fig)