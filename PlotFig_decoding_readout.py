"""
Created: March 30 2016
author: gabeo

Figure: population discrimination of two stimuli from:
    linear noise (tree level mean and two-point)
    second order (one-loop mean and two-point)
    consistent second-order (one-loop mean, tree-level two-point)

Steps for calculation:
    pick synaptic weight
    pick two stimuli
    for each case:
        calculate rates and correlations for each stimulus
        compute log odds ratio
"""

''' Import libraries '''
import params
reload(params)
from generate_adj import generate_adj as gen_adj
import sim_poisson
import numpy as np
import matplotlib.pyplot as plt
from raster import raster
from theory import rates_ss
from theory import rates_1loop
from theory import two_point_function_fourier
from theory import two_point_function_fourier_1loop
import time
import os
import sys
import math

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

''' simulation parameters '''

tstop = 50. * tau
dt = .02 * tau
trans = 5. * tau

syn_scale = 70 # 10  #

''' set save directory '''
if sys.platform == 'darwin':
    save_dir = '/Users/gabeo/Documents/projects/structure_driven_activity/decoding_Ne=200_readout/'
elif sys.platform == 'linux2':
    save_dir = '/home/gabeo/Documents/projects/structure_driven_activity/decoding_Ne=200_readout/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

''' load or generate adjacency matrix '''
W0_path = os.path.join(save_dir, 'W0.npy')

if os.path.exists(W0_path):

    W0 = np.load(os.path.join(save_dir, 'W0.npy'))

else:

    W0 = gen_adj(Ne, Ni, pEE, pEI, pIE, pII)

    if Ne > 0:  # make first neuron a readout
        W0[0, :] = 0
        W0[0, 1:Ne] = 1
        W0[:, 0] = 0

    savefile = os.path.join(save_dir, 'W0.npy')
    np.save(savefile, W0)

stim_1_set = raw_input("Enter 'True' after setting drive for stim 1 in params.py")

if stim_1_set:

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale

    r_th_0loop_stim1 = rates_ss(W)
    two_point_0loop_stim1 = two_point_function_fourier(W)[0].reshape((N,N)).real

    r_th_1loop_stim1 = r_th_0loop_stim1 + rates_1loop(W)
    two_point_1loop_stim1 = two_point_0loop_stim1 + two_point_function_fourier_1loop(W)[0].reshape((N,N)).real

    ''' generate realization of activity for stim 1'''
    spktimes = sim_poisson.sim_poisson(W, tstop, trans, dt)
    spktimes[:, 0] -= trans

    print 'plotting raster'
    savefile = os.path.join(save_dir, 'raster_stim1.pdf')
    raster(spktimes, spktimes.shape[0], savefile)

    # spktimes = sim_poisson.sim_poisson(W, tstop, trans, dt)
    # spktimes[:,0] -= trans
    r_sim_stim1 = np.zeros(N)
    for n in range(N):
        r_sim_stim1[n] = sum(spktimes[:, 1] == n) / (tstop - trans)

    print 'mean rate for stim 1 (sp/ms): ', np.mean(r_sim_stim1[:Ne])
    print 'readout rate for stim 1 (sp/ms): ', r_sim_stim1[0]

else:
    raise Exception("Please re-run, set drive for stim 1 and don't mis-spell 'True' ")

stim_2_set = raw_input("Enter 'True' after setting drive for stim 2 in params.py")

if stim_2_set:

    ### generate scaled weight matrix from frozen connectivity realization
    W = W0 * 1.
    if Ne > 0:
        W[0:Ne, 0:Ne] = weightEE * W0[0:Ne, 0:Ne]
        W[Ne:, 0:Ne] = weightIE * W0[Ne:, 0:Ne]

    if Ni > 0:
        W[0:Ne, Ne:] = weightEI * W0[0:Ne, Ne:]
        W[Ne:, Ne:] = weightII * W0[Ne:, Ne:]

    W *= syn_scale

    r_th_0loop_stim2 = rates_ss(W)
    two_point_0loop_stim2 = two_point_function_fourier(W)[0].reshape((N,N)).real

    r_th_1loop_stim2 = r_th_0loop_stim2 + rates_1loop(W)
    two_point_1loop_stim2 = two_point_0loop_stim2 + two_point_function_fourier_1loop(W)[0].reshape((N,N)).real

    ''' generate realization of activity for stim 1'''
    spktimes = sim_poisson.sim_poisson(W, 1000*tau, trans, dt)
    spktimes[:, 0] -= trans
    print 'plotting raster'
    savefile = os.path.join(save_dir, 'raster_stim2.pdf')
    raster(spktimes, spktimes.shape[0], savefile)

    # spktimes = sim_poisson.sim_poisson(W, tstop, trans, dt)
    # spktimes[:, 0] -= trans
    r_sim_stim2 = np.zeros(N)
    for n in range(N):
        r_sim_stim2[n] = sum(spktimes[:, 1] == n) / (tstop - trans)

    print 'mean rate for stim 2 (sp/ms): ', np.mean(r_sim_stim2[:Ne])
    print 'readout rate for stim 2 (sp/ms): ', r_sim_stim2[0]

else:
    raise Exception("Please re-run, set drive for stim 2 and don't mis-spell 'True' ")

''' compute log odds for each stimulus in three cases: tree level, one-loop level and consistent one-loop level '''
p_stim1 = 0.5
p_stim2 = 0.5

N_decode = 1

print two_point_0loop_stim1.shape
print two_point_0loop_stim1.dtype

if N_decode > 1:
    x_mu_T = (r_sim_stim1[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim1[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x1_stim1_0loop = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_0loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_0loop_stim1[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim1[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim1[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x1_stim1_1loop_consistent = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_0loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_0loop_stim1[:N_decode, :N_decode])), x_mu))

    p_x1_stim1_1loop = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_1loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(-.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(
        two_point_1loop_stim1[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x1_stim2_0loop = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
    two_point_0loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(-.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(
        two_point_0loop_stim2[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x1_stim2_1loop_consistent = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_0loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(-.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(
        two_point_0loop_stim2[:N_decode, :N_decode])), x_mu))

    p_x1_stim2_1loop = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_1loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(-.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(
        two_point_1loop_stim2[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x2_stim1_0loop = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_0loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(-.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(
        two_point_0loop_stim1[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x2_stim1_1loop_consistent = np.sqrt((2 * math.pi) ** N_decode * np.linalg.det(
        two_point_0loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_0loop_stim1[:N_decode, :N_decode])), x_mu))

    p_x2_stim1_1loop = np.sqrt(
        (2 * math.pi) ** N_decode * np.linalg.det(two_point_1loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_1loop_stim1[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x2_stim2_0loop = np.sqrt(
        (2 * math.pi) ** N_decode * np.linalg.det(two_point_0loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_0loop_stim2[:N_decode, :N_decode])), x_mu))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x2_stim2_1loop_consistent = np.sqrt(
        (2 * math.pi) ** N_decode * np.linalg.det(two_point_0loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_0loop_stim2[:N_decode, :N_decode])), x_mu))

    p_x2_stim2_1loop = np.sqrt(
        (2 * math.pi) ** N_decode * np.linalg.det(two_point_1loop_stim2[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * np.dot(np.dot(x_mu_T, np.linalg.inv(two_point_1loop_stim2[:N_decode, :N_decode])), x_mu))

else: # N_decode = 1

    x_mu_T = (r_sim_stim1[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim1[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x1_stim1_0loop = np.sqrt((2 * math.pi) ** N_decode * two_point_0loop_stim1[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / (two_point_0loop_stim1[:N_decode, :N_decode]))

    x_mu_T = (r_sim_stim1[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim1[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x1_stim1_1loop_consistent = np.sqrt(
        (2 * math.pi) ** N_decode * (two_point_0loop_stim1[:N_decode, :N_decode])) ** -1 * np.exp(
        -.5 * x_mu ** 2 / (two_point_0loop_stim1[:N_decode, :N_decode]))

    p_x1_stim1_1loop = np.sqrt((2 * math.pi) ** N_decode * two_point_1loop_stim1[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / (two_point_1loop_stim1[:N_decode, :N_decode]))

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x1_stim2_0loop = np.sqrt((2 * math.pi) ** N_decode * two_point_0loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 /
        two_point_0loop_stim2[:N_decode, :N_decode])

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x1_stim2_1loop_consistent = np.sqrt((2 * math.pi) ** N_decode *
                                          two_point_0loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(-.5 * x_mu ** 2 /
                                                                                                      two_point_0loop_stim2[
                                                                                                      :N_decode,
                                                                                                      :N_decode])

    p_x1_stim2_1loop = np.sqrt((2 * math.pi) ** N_decode *
                               two_point_1loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(-.5 * x_mu ** 2 /
                                                                                           two_point_1loop_stim2[
                                                                                           :N_decode, :N_decode])

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x2_stim1_0loop = np.sqrt((2 * math.pi) ** N_decode *
                               two_point_0loop_stim1[:N_decode, :N_decode]) ** -1 * np.exp(-.5 * x_mu ** 2 /
                                                                                           two_point_0loop_stim1[
                                                                                           :N_decode, :N_decode])

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim1[:N_decode]).reshape((N_decode, 1))

    p_x2_stim1_1loop_consistent = np.sqrt((2 * math.pi) ** N_decode *
                                          two_point_0loop_stim1[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / two_point_0loop_stim1[:N_decode, :N_decode])

    p_x2_stim1_1loop = np.sqrt(
        (2 * math.pi) ** N_decode * two_point_1loop_stim1[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / two_point_1loop_stim1[:N_decode, :N_decode])

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_0loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x2_stim2_0loop = np.sqrt(
        (2 * math.pi) ** N_decode * two_point_0loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / two_point_0loop_stim2[:N_decode, :N_decode])

    x_mu_T = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((1, N_decode))
    x_mu = (r_sim_stim2[:N_decode] - r_th_1loop_stim2[:N_decode]).reshape((N_decode, 1))

    p_x2_stim2_1loop_consistent = np.sqrt(
        (2 * math.pi) ** N_decode * two_point_0loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / two_point_0loop_stim2[:N_decode, :N_decode])

    p_x2_stim2_1loop = np.sqrt(
        (2 * math.pi) ** N_decode * two_point_1loop_stim2[:N_decode, :N_decode]) ** -1 * np.exp(
        -.5 * x_mu ** 2 / two_point_1loop_stim2[:N_decode, :N_decode])

p_stim1_x1_0loop = p_x1_stim1_0loop * p_stim1 / (p_x1_stim1_0loop * p_stim1 + p_x1_stim2_0loop * p_stim2)
p_stim1_x1_1loop_consistent = p_x1_stim1_1loop_consistent * p_stim1 / (
p_x1_stim1_1loop_consistent * p_stim1 + p_x1_stim2_1loop_consistent * p_stim2)
p_stim1_x1_1loop = p_x1_stim1_1loop * p_stim1 / (p_x1_stim1_1loop * p_stim1 + p_x1_stim2_1loop * p_stim2)

p_stim1_x2_0loop = p_x2_stim1_0loop * p_stim1 / (p_x2_stim1_0loop * p_stim1 + p_x2_stim2_0loop * p_stim2)
p_stim1_x2_1loop_consistent = p_x2_stim1_1loop_consistent * p_stim1 / (
p_x2_stim1_1loop_consistent * p_stim1 + p_x2_stim2_1loop_consistent * p_stim2)
p_stim1_x2_1loop = p_x2_stim1_1loop * p_stim1 / (p_x2_stim1_1loop * p_stim1 + p_x2_stim2_1loop * p_stim2)

p_stim2_x1_0loop = p_x1_stim2_0loop * p_stim2 / (p_x1_stim1_0loop * p_stim1 + p_x1_stim2_0loop * p_stim2)
p_stim2_x1_1loop_consistent = p_x1_stim2_1loop_consistent * p_stim2 / (
p_x1_stim1_1loop_consistent * p_stim1 + p_x1_stim2_1loop_consistent * p_stim2)
p_stim2_x1_1loop = p_x1_stim2_1loop * p_stim2 / (p_x1_stim1_1loop * p_stim1 + p_x1_stim2_1loop * p_stim2)

p_stim2_x2_0loop = p_x2_stim2_0loop * p_stim2 / (p_x2_stim1_0loop * p_stim1 + p_x2_stim2_0loop * p_stim2)
p_stim2_x2_1loop_consistent = p_x2_stim2_1loop_consistent * p_stim2 / (
p_x2_stim1_1loop_consistent * p_stim1 + p_x2_stim2_1loop_consistent * p_stim2)
p_stim2_x2_1loop = p_x2_stim2_1loop * p_stim2 / (p_x2_stim1_1loop * p_stim1 + p_x2_stim2_1loop * p_stim2)


log_odds_0loop_stim1 = np.log(p_stim1_x1_0loop / p_stim2_x1_0loop)
log_odds_1loop_consistent_stim1 = np.log(p_stim1_x1_1loop_consistent / p_stim2_x1_1loop_consistent)
log_odds_1loop_stim1 = np.log(p_stim1_x1_1loop / p_stim2_x1_1loop)

log_odds_0loop_stim2 = np.log(p_stim2_x2_0loop / p_stim1_x2_0loop)
log_odds_1loop_consistent_stim2 = np.log(p_stim2_x2_1loop_consistent / p_stim1_x2_1loop_consistent)
log_odds_1loop_stim2 = np.log(p_stim2_x2_1loop / p_stim1_x2_1loop)

print log_odds_0loop_stim1
print log_odds_1loop_consistent_stim1
print log_odds_1loop_stim1
print log_odds_0loop_stim2
print log_odds_1loop_consistent_stim2
print log_odds_1loop_stim2

fig, ax = plt.subplots(1)
bar1 = ax.bar(np.arange(3), np.array([log_odds_0loop_stim1[0][0], log_odds_1loop_consistent_stim1[0][0], log_odds_1loop_stim1[0][0]]), width=0.5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(("0 loop", "1 loop rates", "1 loop rates and correlations"))
savefile = os.path.join(save_dir, 'log_odds_stim1.pdf')
plt.savefig(savefile)
plt.show()
plt.close()

#plt.xticklabels('0 loop', '1 loop for rates', '1 loop')