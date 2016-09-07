# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:04:49 2015

@author: gabeo

simulate a network of Poisson neurons with alpha-function synapses
"""
def sim_poisson(W, tstop, trans, dt):
    import os
    import numpy as np
    from phi import phi
    from phi import phi_prime
    from theory import rates_ss
    import params; reload(params)
    from generate_adj import generate_adj as gen_adj
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as plt3
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    # unpackage parameters
    par = params.params()
    
#    Ne = par.Ne
#    Ni = par.Ni
    N = par.N
#    pEE = par.pEE
#    pEI = par.pEI
#    pIE = par.pIE
#    pII = par.pII
#    weightEE = par.weightEE
#    weightEI = par.weightEI
#    weightIE = par.weightIE
#    weightII = par.weightII
    tau = par.tau
    b = par.b
    gain = par.gain
    
    ## generate adjacency matrix
    #W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)
    ## make first neuron a readout
    #W0[0,1:] = 1
    #W0[:,0] = 0
    
    # generate weight matrix
#    W = np.dot(W0, np.eye(N))
#    if Ne > 0:
#        W[0:Ne,0:Ne] = weightEE*W0[0:Ne,0:Ne]
#        W[Ne:,0:Ne] = weightIE*W0[Ne:,0:Ne]
#        
#    if Ni > 0:
#        W[0:Ne,Ne:] = weightEI*W0[0:Ne,Ne:]
#        W[Ne:,Ne:] = weightII*W0[Ne:,Ne:]
#    
#    r_th = rates_ss(W)    
#    g = np.dot(W,r_th) + b
#    stab_matrix = np.dot(np.diag(phi_prime(g, gain)), W)
#    spec_rad_actual = max(abs(np.linalg.eigvals(stab_matrix)))
#    #spec_rad_desired = .8 # .2 and .8?
#    
#    W *= spec_rad_desired/spec_rad_actual
    
#    print np.mean(W, axis=(0,1))
    
#    dt = .02*tau
#    trans = 5.*tau
#     tstop += trans
#    tstop = 10000.*tau + trans
    Nt = int(tstop/dt)
#    Ntrans = int(trans/dt)
    t = 0
    numspikes = 0
    
    count = np.zeros((N,)) # spike count of each neuron in the interval (0,tstop)
    maxspikes = 500*N*tstop/1000  # 100 Hz / neuron
    spktimes = np.zeros((maxspikes,2)) # spike times and neuron labels

    g_vec = np.zeros((Nt, N))

#    Tmax = 10.*tau
    
    s = np.zeros((N,)) # synaptic output of each neuron
    s0 = np.zeros((N,))
    
    s_dummy = np.zeros((N,))
    
    a = 1. / tau
    a2 = a**2
    
    for i in range(0,Nt,1):
        
        t += dt
        
        # update each neuron's output and plasticity traces
        s_dummy += dt*(-2*a*s_dummy - a2*s)    
        s = s0 + dt*s_dummy
        
        # compute each neuron's input
        g = np.dot(W, s) + b
        g_vec[i] = g

        # decide if each neuron spikes, update output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain)
        
        # spiket = np.random.rand(N,)
        # spiket[spiket<r*dt] = 1
        # spiket[spiket!=1] = 0

        # print np.amax(r)

        try:
            spiket = np.random.poisson(r*dt, size=(N,))
        except:
            break

        s_dummy += spiket*a2
    
        ### store spike times and counts
        if t > trans:
            count += spiket
            for j in range(0,N):
                if spiket[j] >= 1 and numspikes < maxspikes:
                    spktimes[numspikes, 0] = t
                    spktimes[numspikes, 1] = j
                    numspikes += 1
        
        s0 = s
    
    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]
    return spktimes, g_vec
    
    #from correlation_functions import bin_spiketrain
    #spk_post = bin_spiketrain(spktimes,0,dt,1,tstop,trans)
    #
    #print sum(spk_post)/(tstop-trans)*1000
    #count_post = pop.count_dist(spk_post, 1 , tstop, 100)
    #plt.figure(); plt.hist(count_post,bins=10)
    #plt.title('Postsynaptic spike count distribution')
    
    ### exmpirical convariance
    #from correlation_functions import cross_covariance_spk
    #print 'computing second cumulants (empirical)'
    #
    #lags = np.arange(-Tmax,Tmax+1,1)
    #Nlags = lags.size
    #xcov2 = np.zeros((N,N,Nlags))
    #for i in range(N):
    #    for j in range(N):
    #        xcov2[i,j,:] = cross_covariance_spk(spktimes,numspikes,i,j,dt,lags,tau,tstop,trans)
    #
    #plt.figure(); plt.plot(lags, xcov2[0,1,:])
                    
    ### plot raster
    #from raster import raster
    #tplot = 5000
    #spktimes_plot = spktimes[np.where(spktimes[:,0]<tplot)[0],:]
    #numspikes_plot = np.shape(spktimes_plot)[0]
    #raster(spktimes_plot,numspikes_plot)
    
    ### total triplet autocorrelation
    #print 'computing total triplet autocovariances (empirical)'
    #from correlation_functions import triplet_covariance_tot
    #from correlation_functions import cross_covariance_tot
    #lags = np.arange(-Tmax,Tmax+1,dt_ccg)
    #autocov2_tot = np.zeros((N,))
    #autocov3_tot = np.zeros((N,))
    #for n in range(N):
    #    autocov2_tot[n] = cross_covariance_tot(spktimes,numspikes,n,n,dt,lags,tau,tstop,trans)
    #    autocov3_tot[n] = triplet_covariance_tot(spktimes,numspikes,n,n,n,dt,lags,tau,tstop,trans)
    
    
    ### third cumulants
    #print 'computing third cumulants (empirical)'
    #from correlation_functions import triplet_covariance_spk
    #lags = np.arange(-Tmax,Tmax+1,1)
    #Nlags = lags.size
    #ind_calc = np.array((0,1,10,15))
    #Ncalc = ind_calc.size
    #xcov3 = np.zeros((N,N,N,Nlags,Nlags))
    #for i in range(Ncalc):
    #    for j in range(Ncalc):
    #        for k in range(Ncalc):
    #                xcov3[i,j,k,:,:] = triplet_covariance_spk(spktimes,numspikes,i,j,k,dt,lags,tau,tstop,trans)
    ###           1
    #ind0 = np.where(np.abs(lags)==min(np.abs(lags)))[0][0]
    #plt.figure(); 
    #plt.plot(lags,xcov3[0,1,0,:,ind0],'o')
    #
    #plt.figure(); 
    #plt.plot(lags,xcov3[1,0,1,:,ind0],'o')
    #
    #plt.figure();
    #plt.plot(lags,xcov3[0,1,1,ind0,:],'o')
    #
    #fig = plt.figure()
    #ax = plt3.Axes3D(fig)
    #X, Y = np.meshgrid(lags, lags)
    #ax.plot_surface(X,Y,xcov3[0,1,0,:,:])
    #ax.set_ylabel('Time lag, s1 (ms)')
    #ax.set_xlabel('Time lag, s2 (ms)')
    #ax.set_zlabel('Third cross-cumulant, cells 0,1,0 (sp/ms)'+r'$^3$')
    #ax.set_zlim((0,.000121))
    
    #save_dir = '/home/gabeo/Documents/projects/structure_driven_activity/sims_N='+str(N)
    #savefile = os.path.join(save_dir,'xcov2.npy')
    #np.save(savefile,xcov2)
    #
    #savefile = os.path.join(save_dir,'xcov3.npy')
    #np.save(savefile,xcov3)
    #
    #savefile = os.path.join(save_dir,'W.npy')
    #np.save(savefile,W)
    #
    #savefile = os.path.join(save_dir,'spktimes.npy')
    #np.save(savefile,spktimes)

if __name__=='__main__':
    
    par = params.params()
    
    Ne = par.Ne
    Ni = par.Ni
    N = par.N
    pEE = par.pEE
    pEI = par.pEI
    pIE = par.pIE
    pII = par.pII
    weightEE = par.weightEE
    weightEI = par.weightEI
    weightIE = par.weightIE
    weightII = par.weightII
    tau = par.tau
    b = par.b
    gain = par.gain
    
    # generate adjacency matrix
    W0 = gen_adj(Ne,Ni,pEE,pEI,pIE,pII)
    # make first neuron a readout
    W0[0,1:] = 1
    W0[:,0] = 0
    
    # generate weight matrix
    W = np.dot(W0, np.eye(N))
    if Ne > 0:
        W[0:Ne,0:Ne] = weightEE*W0[0:Ne,0:Ne]
        W[Ne:,0:Ne] = weightIE*W0[Ne:,0:Ne]
        
    if Ni > 0:
        W[0:Ne,Ne:] = weightEI*W0[0:Ne,Ne:]
        W[Ne:,Ne:] = weightII*W0[Ne:,Ne:]
        
    sim_poisson(W, tstop, trans, dt)