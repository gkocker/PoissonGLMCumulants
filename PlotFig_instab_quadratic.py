'''

plot the instability figure with loaded sims / theory, and using gridspec to set up figure
'''


''' Import libraries '''
import params; reload(params)
from generate_adj import generate_adj as gen_adj

import sim_poisson
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from raster import raster
from correlation_functions import auto_covariance_pop
from correlation_functions import bin_pop_spiketrain
from correlation_functions import cross_covariance_spk
from phi import phi_prime
from phi import phi
from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier_pop
from theory import two_point_function_fourier_pop_1loop
from theory import stability_matrix_1loop

import time
import os
import sys

''' set up subplot grid '''
plt.figure(figsize=(7.5, 4))
gs1 = gridspec.GridSpec(2, 3)
gs1.update(top=.95, bottom=.6, hspace=0.1, left=0.2)
ax1 = plt.subplot(gs1[:2, 0])
ax2 = plt.subplot(gs1[:2, 1])
ax3 = plt.subplot(gs1[:2, 2])

gs2 = gridspec.GridSpec(3, 1)
gs2.update(top=.5, bottom=.05, hspace=0.1, left=0.2)
ax4 = plt.subplot(gs2[0, :])
ax5 = plt.subplot(gs2[1, :])
ax6 = plt.subplot(gs2[2, :])



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

tstop = 100. * tau
Ncalc = 3

dt = .02 * tau
trans = 5. * tau
window = tstop
Tmax = 8 * tau
dt_ccg = 1
lags = np.arange(-Tmax, Tmax, dt_ccg)

# syn_scale = np.array((1., 57.)) # for quadratic, was 75
syn_scale = np.array((1., 60.))
# syn_scale = np.array((0., 1., 12.))  # for linear


''' set save directory '''
if sys.platform == 'darwin': save_dir = '/Users/gabeo/Documents/projects/field_theory_spiking/1loop_Ne=200_softplus/'
elif sys.platform == 'linux2': save_dir = '/local1/Documents/projects/structure_driven_activity/1loop_Ne=200_softplus/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

''' load or generate adjacency matrix '''
W0_path = os.path.join(save_dir, 'W0.npy')

if os.path.exists(W0_path):
    W0 = np.load(os.path.join(save_dir, 'W0.npy'))
else:
    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)

    if Ne > 0: # make first neuron a readout
        W0[0, :] = 0
        W0[0, 1:Ne] = 1
        W0[:, 0] = 0

    savefile = os.path.join(save_dir, 'W0.npy')
    np.save(savefile, W0)

calc_range = range(Ncalc)


'''
plot transfer function
'''
x = np.arange(-1, 1, .01)
y = gain*x**2
y[np.where(x<0)[0]] = 0
ax1.plot(x, y, 'k', linewidth=2)
ax1.set_xlabel('Membrane voltage', fontsize=12)
ax1.set_ylabel('Firing rate', fontsize=12)
ax1.set_xlim((-1, 1))
ax1.set_ylim((-.001, .1))
ax1.set_xticks([0])
ax1.set_yticks([])

print 'running sims 1'

### generate scaled weight matrix from frozen connectivity realization
W = W0 * 1.
if Ne > 0:
    W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

if Ni > 0:
    W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

W *= syn_scale[0]

spktimes, g_vec1 = sim_poisson.sim_poisson(W, tstop, trans, dt)

print 'plotting raster 1'
spktimes[:, 0] -= trans
numspikes = spktimes.shape[0]

for i in range(0, numspikes):
    ax2.plot(spktimes[i, 0]/1000., spktimes[i, 1], 'k.', markersize=1)

ax2.set_yticks([0, Ne-1])
ax2.set_xticks([0, tstop/1000.])
ax2.set_ylabel('Neuron', fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylim((0, N))
ax2.set_xlim((0, tstop/1000.))
ax2.set_title('Weak synapses')

print 'running sims 2'
W = W0 * 1.
if Ne > 0:
    W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
    W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

if Ni > 0:
    W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
    W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

W *= syn_scale[1]

spktimes, g_vec2 = sim_poisson.sim_poisson(W, tstop, trans, dt)

print 'plotting raster 2'
spktimes[:, 0] -= trans
numspikes = spktimes.shape[0]

for i in range(0, numspikes):
    ax3.plot(spktimes[i, 0]/1000., spktimes[i, 1], 'k.', markersize=1)

ax3.set_yticks([])
ax3.set_xticks([0, tstop/1000.])
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylim((0, N))
ax3.set_xlim((0, tstop/1000.))
ax3.set_title('Strong synapses')

''' plot sim / theory as synaptic strength scales '''
#
# savefile = os.path.join(save_dir, 'rE_av_sim.npy')
# rE_av_sim = np.load(savefile)
#
# savefile = os.path.join(save_dir, 'two_point_pop_sim.npy')
# two_point_pop_sim = np.load(savefile)
#
# savefile = os.path.join(save_dir, 'two_point_integral_sim.npy')
# two_point_integral_sim = np.load(savefile)

''' compute stability '''
Ncalc = 10  # 3 for rasters

tstop = 4000. * tau
dt = .02 * tau
trans = 5. * tau
window = tstop
Tmax = 8 * tau
dt_ccg = 1
lags = np.arange(-Tmax, Tmax, dt_ccg)

syn_scale = np.linspace(0., 85., Ncalc) # for quadratic
# syn_scale = np.linspace(0., 12., Ncalc)  # for linear

rE_av_theory = np.zeros((Ncalc, ))
rE_av_1loop = np.zeros((Ncalc, ))

spec_rad = np.zeros(Ncalc, )
spec_rad_1loop = np.zeros(Ncalc, )
two_point_integral_theory = np.zeros(Ncalc, )
two_point_integral_1loop = np.zeros(Ncalc, )

spec_rad = np.zeros(Ncalc, )
spec_rad_1loop = np.zeros(Ncalc, )

two_point_pop_sim = np.zeros((Ncalc, len(lags)))
two_point_integral_sim = np.zeros(Ncalc, )
rE_av_sim = np.zeros(Ncalc, )

print 'ranging over synaptic weights'
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
    r_th = rates_ss(W)
    r_th_11oop = rates_1loop(W)

    rE_av_theory[nn] = np.mean(r_th[1:Ne]).real
    rE_av_1loop[nn] = np.mean(r_th_11oop[1:Ne]).real

    g = np.dot(W, r_th) + b
    w = 1e-6
    stab_mat_mft = np.dot(np.diag(phi_prime(g, gain)), W)
    stab_mat_1loop = stability_matrix_1loop(w, W, r_th)
    spec_rad[nn] = max(abs(np.linalg.eigvals(stab_mat_mft)))
    spec_rad_1loop[nn] = max(abs(np.linalg.eigvals(stab_mat_mft - stab_mat_1loop)))

    two_point_integral_theory[nn] = np.real(two_point_function_fourier_pop(W, range(Ne))[0])
    two_point_integral_1loop[nn] = np.real(two_point_function_fourier_pop_1loop(W, range(Ne))[0])

    spktimes, g_vec = sim_poisson.sim_poisson(W, tstop, trans, dt)
    spktimes[:, 0] -= trans

    ind_include = range(Ne)
    spk_Epop = bin_pop_spiketrain(spktimes, dt, 1, tstop, trans, ind_include)
    rE_av_sim[nn] = np.sum(spk_Epop) / float(tstop-trans) / float(len(ind_include))

    two_point_pop_sim[nn, :] = auto_covariance_pop(spktimes, range(Ne), spktimes.shape[0], dt, lags, tau,
                                                       tstop, trans)
    two_point_integral_sim[nn] = np.sum(two_point_pop_sim[nn, :]) * 1

Nstab = Ncalc
fac10 = np.floor(np.log10(1.5*np.amax(rE_av_theory[:Nstab]+rE_av_1loop[:Nstab]))).astype(int)

rE_av_sim *= 1000
rE_av_theory *= 1000
rE_av_1loop *= 1000
ax4.plot(syn_scale*weightEE, rE_av_sim, 'ko')
ax4.plot(syn_scale[:Nstab]*weightEE, rE_av_theory[:Nstab], 'k', linewidth=2)
ax4.plot(syn_scale[:Nstab]*weightEE, rE_av_theory[:Nstab]+rE_av_1loop[:Nstab], 'r', linewidth=2)
ax4.set_yticks(np.linspace(np.round(.9*np.amin(rE_av_theory[:Nstab]), decimals=4), np.round(1.5*np.amax(rE_av_theory[:Nstab]+rE_av_1loop[:Nstab]), decimals=3), num=3))
ax4.set_ylim((np.round(.9*np.amin(rE_av_theory[:Nstab]), decimals=4), np.round(1.2*np.amax(rE_av_theory[:Nstab]+rE_av_1loop[:Nstab]), decimals=3)))
ax4.set_xlim((0, np.amax(syn_scale*weightEE)))
ax4.set_ylabel('Rate \n (sp/s)', fontsize=12)

two_point_integral_sim *= 1000
two_point_integral_theory *= 1000
two_point_integral_1loop *= 1000

ax5.plot(syn_scale*weightEE, two_point_integral_sim, 'ko')
ax5.plot(syn_scale[:Nstab]*weightEE, two_point_integral_theory[:Nstab], 'k', linewidth=2)
ax5.plot(syn_scale[:Nstab]*weightEE, two_point_integral_theory[:Nstab]+two_point_integral_1loop[:Nstab], 'r', linewidth=2)
ax5.set_yticks(np.linspace(np.round(.9*np.amin(two_point_integral_theory[:Nstab]), decimals=3), np.round(1*np.amax(two_point_integral_theory[:Nstab]+two_point_integral_1loop[:Nstab]), decimals=3), num=3))
ax5.set_ylim((np.round(.9*np.amin(two_point_integral_theory[:Nstab]), decimals=3), np.round(1.2*np.amax(two_point_integral_theory[:Nstab]+two_point_integral_1loop[:Nstab]), decimals=4)))
ax5.set_xlim((0, np.amax(syn_scale*weightEE)))
ax5.set_ylabel('Cov. \n (sp'+r'$^2$'+'/s)', fontsize=12)

ax6.plot(syn_scale[:Nstab]*weightEE, spec_rad[:Nstab], 'k', linewidth=2)
ax6.plot(syn_scale[:Nstab]*weightEE, spec_rad_1loop[:Nstab], 'r', linewidth=2)
ax6.plot(syn_scale[:Nstab]*weightEE, np.ones(Nstab), 'k--')
ax6.set_xlabel('Exc.-exc. synaptic weight (mV)', fontsize=12)
ax6.set_xlim((0, np.amax(syn_scale*weightEE)))
ax6.set_ylabel('Stability', fontsize=12)
ax6.set_yticks([0, 1, 2, 3])

'''
format plots
'''

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')



ax4.set_xticklabels('')
ax5.set_xticklabels('')

# savefile = '/local1/Documents/projects/structure_driven_activity/Fig_instab_quadrati1.eps'
savefile = os.path.join(save_dir, 'fig_instab.pdf')
plt.savefig(savefile)
plt.show()