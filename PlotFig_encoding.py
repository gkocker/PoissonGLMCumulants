'''
Plot figure: mutual information between activity and stimulus for different versions of the theory
params for stim 1: b[:N_ff] = 2*regular
params for sti 2: b[N_ff:2*N_ff] = 2*regular
'''


''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from phi import phi_prime
from phi import phi
from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier
from theory import two_point_function_fourier_1loop
from theory_mutual_information import mutual_inf
from theory_mutual_information import entropy_gaussian

import time
import os
import sys


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/structure_driven_activity/encoding/'
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
        # W0[2*N_ff:Ne, :2*N_ff] =


    savefile = os.path.join(save_dir, 'W0_2layer.npy')
    np.save(savefile, W0)


''' plot mutual information between activity and stimulus as synaptic strength scales '''
Ncalc = 40
N_code = Ne - 2*N_ff

fisher_tree = np.zeros((Ncalc, 2))
fisher_1loop = np.zeros((Ncalc, 2))
fisher_lin_noise = np.zeros((Ncalc, 2))
fisher_quad_noise = np.zeros((Ncalc, 2))

inf_tree = np.zeros((Ncalc, 2))
inf_1loop = np.zeros((Ncalc, 2))
inf_lin_noise = np.zeros((Ncalc, 2))
inf_quad_noise = np.zeros((Ncalc, 2))

condent_tree = np.zeros((Ncalc, 2))
condent_1loop = np.zeros((Ncalc, 2))
condent_lin_noise = np.zeros((Ncalc, 2))
condent_quad_noise = np.zeros((Ncalc, 2))

r_tree_stim1 = np.zeros((Ncalc, N_code))
r_1loop_stim1 = np.zeros((Ncalc, N_code))
cov_tree_stim1 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim1 = np.zeros((Ncalc, N_code, N_code))

r_tree_stim2 = np.zeros((Ncalc, N_code))
r_1loop_stim2 = np.zeros((Ncalc, N_code))
cov_tree_stim2 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim2 = np.zeros((Ncalc, N_code, N_code))

''' calculate theory for stim 1 '''
print 'ranging over synaptic weights for linear network, stim 1'

phi_set = raw_input("Enter 'True' after checking that transfer is threshold-linear in phi.py")
b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")
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
    r_tree_stim1[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1000
    r_1loop_stim1[nn, :] = r_tree_stim1[nn, :] + rates_1loop(W)[2*N_ff:2*N_ff+N_code]*1000

    cov_tree_stim1[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000
    cov_1loop_stim1[nn, :, :] = cov_tree_stim1[nn, :, :] + np.real(two_point_function_fourier_1loop(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000

''' calculate theory for stim 2 and mutual inf '''
b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")
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
    r_tree_stim2[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1000
    r_1loop_stim2[nn, :] = r_tree_stim2[nn, :] + rates_1loop(W)[2*N_ff:2*N_ff+N_code]*1000

    cov_tree_stim2[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000
    cov_1loop_stim2[nn, :, :] = cov_tree_stim2[nn, :, :] + np.real(two_point_function_fourier_1loop(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000

    u1, v1 = np.linalg.eig(np.diag(np.diag(cov_tree_stim1[nn, : :])))
    u2, v2 = np.linalg.eig(np.diag(np.diag(cov_tree_stim2[nn, :, :])))
    ind1 = np.where(u1 == 0)[0]
    ind2 = np.where(u2 == 0)[0]
    if len(ind1) == 0 and len(ind2) == 0:
        inf_tree[nn, 0] = mutual_inf(r_tree_stim1[nn, :], r_tree_stim2[nn, :], np.diag(np.diag(cov_tree_stim1[nn, :, :])), np.diag(np.diag(cov_tree_stim2[nn, :, :])))

    u1, v1 = np.linalg.eig(cov_tree_stim1[nn, : :])
    u2, v2 = np.linalg.eig(cov_tree_stim2[nn, :, :])
    ind1 = np.where(u1 == 0)[0]
    ind2 = np.where(u2 == 0)[0]
    if len(ind1) == 0 and len(ind2) == 0:
        inf_1loop[nn, 0] = mutual_inf(r_1loop_stim1[nn, :], r_1loop_stim2[nn, :], cov_tree_stim1[nn, :, :], cov_tree_stim2[nn, :, :], p_stim1=0.5, p_stim2=0.5)
        inf_lin_noise[nn, 0] = mutual_inf(r_tree_stim1[nn, :], r_tree_stim2[nn, :], cov_tree_stim1[nn, :, :], cov_tree_stim2[nn, :, :], p_stim1=0.5, p_stim2=0.5)

    u1, v1 = np.linalg.eig(cov_1loop_stim1[nn, : :])
    u2, v2 = np.linalg.eig(cov_1loop_stim2[nn, :, :])
    ind1 = np.where(u1 == 0)[0]
    ind2 = np.where(u2 == 0)[0]
    if len(ind1) == 0 and len(ind2) == 0:
        inf_quad_noise[nn, 0] = mutual_inf(r_1loop_stim1[nn, :], r_1loop_stim2[nn, :], cov_1loop_stim1[nn, :, :], cov_1loop_stim2[nn, :, :], p_stim1=0.5, p_stim2=0.5)

    condent_tree[nn, 0] = .5 * entropy_gaussian(r_tree_stim1[nn, :], np.diag(np.diag(cov_tree_stim1[nn, :, :]))) + .5 * entropy_gaussian(r_tree_stim2[nn, :], np.diag(np.diag(cov_tree_stim2[nn, :, :])))
    condent_1loop[nn, 0] = .5 * entropy_gaussian(r_1loop_stim1[nn, :], cov_tree_stim1[nn, :, :]) + .5 * entropy_gaussian(r_1loop_stim2[nn, :], cov_tree_stim2[nn, :, :])
    condent_lin_noise[nn, 0] = .5 * entropy_gaussian(r_tree_stim1[nn, :], cov_tree_stim1[nn, :, :]) + .5 * entropy_gaussian(r_tree_stim2[nn, :], cov_tree_stim2[nn, :, :])
    condent_quad_noise[nn, 0] = .5 * entropy_gaussian(r_1loop_stim1[nn, :], cov_1loop_stim1[nn, :, :]) + .5 * entropy_gaussian(r_1loop_stim2[nn, :], cov_1loop_stim2[nn, :, :])


''' set up subplot grid '''
plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(2, 1)
gs1.update(top=.95, bottom=.05, hspace=0.1, left=0.05, right=0.25)
ax1 = plt.subplot(gs1[0, 0])
ax2 = plt.subplot(gs1[1, 0])

gs2 = gridspec.GridSpec(2, 1)
gs2.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax3 = plt.subplot(gs2[0, 0])
ax4 = plt.subplot(gs2[1, 0])

plt.figure(figsize=(7.5, 4))
gs3 = gridspec.GridSpec(2, 1)
gs3.update(top=.95, bottom=.05, hspace=0.1, left=0.35, right=0.95)
ax5 = plt.subplot(gs2[0, 0])
ax6 = plt.subplot(gs2[1, 0])


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
syn_scale = np.linspace(0., 12., Ncalc)  # for linear
ax3.plot(syn_scale*weightEE, inf_tree[:, 0], label='Tree')
ax3.plot(syn_scale*weightEE, inf_1loop[:, 0], label='1 loop')
ax3.plot(syn_scale*weightEE, inf_lin_noise[:, 0], label='Linear Noise')
ax3.plot(syn_scale*weightEE, inf_quad_noise[:, 0], label='Quadratic Noise')
ax3.plot([syn_scale[Nstab]*weightEE, syn_scale[Nstab]*weightEE], [.8*np.amin(inf_tree[:, 0]), 1.2*np.amax(inf_tree[:, 0])], 'k')


ax5.plot(syn_scale*weightEE, condent_tree[:, 0], label='Tree')
ax5.plot(syn_scale*weightEE, condent_1loop[:, 0], label='1 loop')
ax5.plot(syn_scale*weightEE, condent_lin_noise[:, 0], label='Linear Noise')
ax5.plot(syn_scale*weightEE, condent_quad_noise[:, 0], label='Quadratic Noise')
ax5.plot([syn_scale[Nstab]*weightEE, syn_scale[Nstab]*weightEE], [.8*np.amin(condent_tree[:, 0]), 1.2*np.amax(condent_tree[:, 0])], 'k')



print 'ranging over synaptic weights for nonlinear network, stim 1'
syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
phi_set = raw_input("Enter 'True' after checking that transfer is threshold-quadratic in phi.py")
b_set = raw_input("Enter 'True' after setting b for stim 1 in params.py")
reload(params)

r_tree_stim1 = np.zeros((Ncalc, N_code))
r_1loop_stim1 = np.zeros((Ncalc, N_code))
cov_tree_stim1 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim1 = np.zeros((Ncalc, N_code, N_code))

r_tree_stim2 = np.zeros((Ncalc, N_code))
r_1loop_stim2 = np.zeros((Ncalc, N_code))
cov_tree_stim2 = np.zeros((Ncalc, N_code, N_code))
cov_1loop_stim2 = np.zeros((Ncalc, N_code, N_code))

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
    r_tree_stim1[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1000
    r_1loop_stim1[nn, :] = r_tree_stim1[nn, :] + rates_1loop(W)[2*N_ff:2*N_ff+N_code]*1000

    cov_tree_stim1[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000
    cov_1loop_stim1[nn, :, :] = cov_tree_stim1[nn, :, :] + np.real(two_point_function_fourier_1loop(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000

''' calculate theory for stim 2 and mutual inf '''
b_set = raw_input("Enter 'True' after setting b for stim 2 in params.py")
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
    r_tree_stim2[nn, :] = rates_ss(W)[2*N_ff:2*N_ff+N_code]*1000
    r_1loop_stim2[nn, :] = r_tree_stim2[nn, :] + rates_1loop(W)[2*N_ff:2*N_ff+N_code]*1000

    cov_tree_stim2[nn, :, :] = np.real(two_point_function_fourier(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000
    cov_1loop_stim2[nn, :, :] = cov_tree_stim2[nn, :, :] + np.real(two_point_function_fourier_1loop(W)[0][2*N_ff:2*N_ff+N_code, 2*N_ff:2*N_ff+N_code].reshape((N_code, N_code)))*1000

    inf_tree[nn, 1] = mutual_inf(r_tree_stim1[nn, :], r_tree_stim2[nn, :], np.diag(np.diag(cov_tree_stim1[nn, :, :])), np.diag(np.diag(cov_tree_stim2[nn, :, :])))
    inf_1loop[nn, 1] = mutual_inf(r_1loop_stim1[nn, :], r_1loop_stim2[nn, :], cov_tree_stim1[nn, :, :], cov_tree_stim2[nn, :, :])
    inf_lin_noise[nn, 1] = mutual_inf(r_tree_stim1[nn, :], r_tree_stim2[nn, :], cov_tree_stim1[nn, :, :], cov_tree_stim2[nn, :, :])
    inf_quad_noise[nn, 1] = mutual_inf(r_1loop_stim1[nn, :], r_1loop_stim2[nn, :], cov_1loop_stim1[nn, :, :], cov_1loop_stim2[nn, :, :])

    condent_tree[nn, 1] = .5 * entropy_gaussian(r_tree_stim1[nn, :], np.diag(np.diag(cov_tree_stim1[nn, :, :]))) + .5 * entropy_gaussian(
        r_tree_stim2[nn, :], np.diag(np.diag(cov_tree_stim2[nn, :, :])))
    condent_1loop[nn, 1] = .5 * entropy_gaussian(r_1loop_stim1[nn, :], cov_tree_stim1[nn, :, :]) + .5 * entropy_gaussian(
        r_1loop_stim2[nn, :], cov_tree_stim2[nn, :, :])
    condent_lin_noise[nn, 1] = .5 * entropy_gaussian(r_tree_stim1[nn, :], cov_tree_stim1[nn, :, :]) + .5 * entropy_gaussian(
        r_tree_stim2[nn, :], cov_tree_stim2[nn, :, :])
    condent_quad_noise[nn, 1] = .5 * entropy_gaussian(r_1loop_stim1[nn, :],cov_1loop_stim1[nn, :, :]) + .5 * entropy_gaussian(
        r_1loop_stim2[nn, :], cov_1loop_stim2[nn, :, :])

Nstab = 38



''' plot nonlinear network '''
syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
ax4.plot(syn_scale*weightEE, inf_tree[:, 1], label='Tree')
ax4.plot(syn_scale*weightEE, inf_1loop[:, 1], label='1 loop')
ax4.plot(syn_scale*weightEE, inf_lin_noise[:, 1], label='Linear Noise')
ax4.plot(syn_scale*weightEE, inf_quad_noise[:, 1], label='Quadratic Noise')

ax6.plot(syn_scale*weightEE, condent_tree[:, 1], label='Tree')
ax6.plot(syn_scale*weightEE, condent_1loop[:, 1], label='1 loop')
ax6.plot(syn_scale*weightEE, condent_lin_noise[:, 1], label='Linear Noise')
ax6.plot(syn_scale*weightEE, condent_quad_noise[:, 1], label='Quadratic Noise')


'''
format plots
'''
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

ax4.set_ylabel('Mutual information, nats')
ax6.set_ylabel('Conditional \n response entropy, nats')
ax3.set_xticklabels('')
ax5.set_xticklabels('')



# savefile = '/local1/Documents/projects/structure_driven_activity/Fig_instab_quadratic.eps'
# plt.savefig(savefile)
# plt.show()