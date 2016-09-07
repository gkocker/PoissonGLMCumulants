import matplotlib.pyplot as plt
import os
import numpy as np
import params; reload(params)
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

Tmax = 8 * tau
dt_ccg = 1.
lags = np.arange(-Tmax, Tmax, dt_ccg)

Ncalc = 40
Nstable = Ncalc-4
syn_scale = np.linspace(0., 6., Ncalc)  # for quadratic E, linear I
# syn_scale *= weightEE
syn_scale = syn_scale**0.5
syn_scale *= weightEE

save_dir = '/local1/Documents/projects/structure_driven_activity/1loop_Ne=200_quadraticforEonly2/'

rE_av_sim = np.load(os.path.join(save_dir, 'rE_av_sim.npy'))
rI_av_sim = np.load(os.path.join(save_dir, 'rI_av_sim.npy'))
rE_av_theory = np.load(os.path.join(save_dir, 'rE_av_theory.npy'))
rE_av_1loop = np.load(os.path.join(save_dir, 'rE_av_1loop.npy'))
rI_av_theory = np.load(os.path.join(save_dir, 'rI_av_theory.npy'))
rI_av_1loop = np.load(os.path.join(save_dir, 'rI_av_1loop.npy'))
two_point_pop_sim = np.load(os.path.join(save_dir, 'two_point_pop_sim.npy'))
two_point_integral_sim = np.load(os.path.join(save_dir, 'two_point_integral_sim.npy'))
two_point_integral_theory = np.load(os.path.join(save_dir, 'two_point_integral_theory.npy'))
two_point_integral_1loop = np.load(os.path.join(save_dir, 'two_point_integral_1loop.npy'))

two_point_integral_I_theory = np.load(os.path.join(save_dir, 'two_point_integral_I_theory.npy'))
two_point_integral_I_1loop = np.load(os.path.join(save_dir, 'two_point_integral_I_1loop.npy'))
two_point_Ipop_sim = np.load(os.path.join(save_dir, 'two_point_Ipop_sim.npy'))
two_point_integral_I_sim = np.sum(two_point_Ipop_sim, axis=-1) * dt_ccg

spec_rad = np.load(os.path.join(save_dir, 'spec_rad.npy'))
spec_rad_1loop = np.load(os.path.join(save_dir, 'spec_rad_1loop.npy'))
spec_rad_E = np.load(os.path.join(save_dir, 'spec_rad_E.npy'))
spec_rad_E_1loop = np.load(os.path.join(save_dir, 'spec_rad_E_1loop.npy'))
spec_rad_I = np.load(os.path.join(save_dir, 'spec_rad_I.npy'))
spec_rad_I_1loop = np.load(os.path.join(save_dir, 'spec_rad_I_1loop.npy'))

size = (2., .8)

fig, ax = plt.subplots(1, figsize=size)
ax.plot(syn_scale[:Nstable], rE_av_theory[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir: ax.plot(syn_scale[:Nstable], rE_av_theory[:Nstable] + rE_av_1loop[:Nstable], 'r',
                                     label='One loop', linewidth=2);
ax.plot(syn_scale[0:Ncalc:4], rE_av_sim[0:Ncalc:4], 'ko', label='Sim')
ax.set_ylabel('E Population Rate (sp/ms)')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
ax.set_ylim((0, rE_av_theory[0] * 1.5))
# #ax.legend(loc=0, fontsize=10)

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
ax.plot(syn_scale[:Nstable], rI_av_theory[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir: ax.plot(syn_scale[:Nstable], rI_av_theory[:Nstable] + rI_av_1loop[:Nstable], 'r',
                                     label='One loop',
                                     linewidth=2);
ax.plot(syn_scale[0:Ncalc:4], rI_av_sim[0:Ncalc:4], 'ko', label='Sim')
ax.set_ylabel('I Population Rate (sp/ms)')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
ax.set_ylim((0, rI_av_theory[0] * 1.5))
#ax.legend(loc=0, fontsize=10)

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
ax.plot(syn_scale[:Nstable], two_point_integral_theory[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir: ax.plot(syn_scale[:Nstable],
                                     two_point_integral_theory[:Nstable] + two_point_integral_1loop[:Nstable], 'r',
                                     label='One loop',
                                     linewidth=2);
ax.plot(syn_scale[0:Ncalc:4], two_point_integral_sim[0:Ncalc:4], 'ko', label='Sim')
ax.set_ylabel('E Pop. Spike Train Variance (sp/ms)^2')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
if 'quadratic' in save_dir:
    ax.set_ylim((0, two_point_integral_theory[0] * 3))
elif 'linear' in save_dir:
    ax.set_ylim((0, np.amax(two_point_integral_theory * 1.5)))
# #ax.legend(loc=0, fontsize=10)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'var_Epop_vs_weight.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(1, figsize=size)
ax.plot(syn_scale[:Nstable], two_point_integral_I_theory[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir: ax.plot(syn_scale[:Nstable],
                                     two_point_integral_I_theory[:Nstable] + two_point_integral_I_1loop[:Nstable], 'r',
                                     label='One loop',
                                     linewidth=2);
ax.plot(syn_scale[0:Ncalc:4], two_point_integral_I_sim[0:Ncalc:4], 'ko', label='Sim')
ax.set_ylabel('I Pop. Spike Train Variance (sp^2/ms)')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
if 'quadratic' in save_dir:
    ax.set_ylim((0, two_point_integral_I_theory[0] * 3))
elif 'linear' in save_dir:
    ax.set_ylim((0, np.amax(two_point_integral_I_theory * 1.5)))
#ax.legend(loc=0, fontsize=10)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'var_Ipop_vs_weight.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(1, figsize=size)
ax.plot(syn_scale[:Nstable], spec_rad[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir:
    ax.plot(syn_scale[:Nstable], spec_rad[:Nstable] + spec_rad_1loop[:Nstable],
            'r', label='One loop', linewidth=2)

    ax.plot(syn_scale, np.ones(syn_scale.shape), 'k--')

ax.set_ylabel('Spectral radius of mean field theory')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
ax.set_ylim((0, 1.5))
# #ax.legend(loc=0, fontsize=10)

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

fig, ax = plt.subplots(1, figsize=size)
ax.plot(syn_scale[:Nstable], spec_rad_E[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir:
    ax.plot(syn_scale[:Nstable], spec_rad_E[:Nstable] + spec_rad_E_1loop[:Nstable],
            'r', label='One loop', linewidth=2)

    ax.plot(syn_scale, np.ones(syn_scale.shape), 'k--')

ax.set_ylabel('Spectral radius of mean field theory')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
ax.set_ylim((0, 1.5))
# #ax.legend(loc=0, fontsize=10)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'stability_vs_weight_Eonly.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(1, figsize=(2, 2))
ax.plot(lags, two_point_pop_sim[4, 0, :], 'k', label=r'$W_{EE} = .05$')
ax.plot(lags, two_point_pop_sim[8, 0, :], 'g', label=r'$W_{EE} = 0.1$')
ax.plot(lags, two_point_pop_sim[10, 0, :], 'r', label=r'$W_{EE} = .13$')
ax.plot(lags, two_point_pop_sim[24, 0, :], 'b', label=r'$W_{EE} = 0.3$')
#ax.legend(loc=0, fontsize=10)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'two_point_sim.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(1, figsize=(2, 2))
ind_norm = np.where(lags == 0)[0]
ax.plot(lags, two_point_pop_sim[4, 0, :] / two_point_pop_sim[4, 0, ind_norm], 'k', label=r'$W_{EE} = .05$')
ax.plot(lags, two_point_pop_sim[8, 0, :] / two_point_pop_sim[8, 0, ind_norm], 'g', label=r'$W_{EE} = 0.1$')
ax.plot(lags, two_point_pop_sim[12, 0, :] / two_point_pop_sim[10, 0, ind_norm], 'r', label=r'$W_{EE} = .15$')
ax.plot(lags, two_point_pop_sim[24, 0, :] / two_point_pop_sim[24, 0, ind_norm], 'b', label=r'$W_{EE} = 0.3$')
#ax.legend(loc=0, fontsize=10)
ax.set_ylim((0, 1.))
ax.set_xlim((-50, 50))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'two_point_sim_norm.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)

fig, ax = plt.subplots(1, figsize=size)
ax.plot(syn_scale[:Nstable], spec_rad_I[:Nstable], 'k', label='Tree level', linewidth=2)
if not 'linear' in save_dir:
    ax.plot(syn_scale[:Nstable], spec_rad_I[:Nstable] + spec_rad_I_1loop[:Nstable],
            'r', label='One loop', linewidth=2)

    ax.plot(syn_scale, np.ones(syn_scale.shape), 'k--')

ax.set_ylabel('Spectral radius of I-only mean field theory')
ax.set_xlabel('Exc-exc Synaptic Weight (mV)')
ax.set_xlim((0, np.amax(syn_scale)))
ax.set_ylim((0, 1.5))
#ax.legend(loc=0, fontsize=10)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
    item.set_fontsize(10)
    item.set_fontname('Arial')
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(8)
    item.set_fontname('Arial')

savefile = os.path.join(save_dir, 'stability_Ionly_vs_weight.pdf')
fig.savefig(savefile)
plt.show(fig)
plt.close(fig)
