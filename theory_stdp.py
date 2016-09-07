# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:01:41 2015

@author: gabeo

functions for computing spike train cumulants and triplet stdp
"""

def rates_ss(W): # set inputs here

    """
    Inputs:
    W, weight matrix
    """

    import numpy as np
    from phi import phi
    import params; reload(params)
    
    par = params.params()    
    b = par.b	
    gain = par.gain

#    W = W*tau
    
    max_fp_its = 100 # maximum number of iterations
    
    r0 = phi(b,gain)    
    
    for i in range(0,max_fp_its):
        g = np.dot(W,r0) + b        
        r = phi(g,gain)
        r0 = r

#    W = W/tau    
    
    return r
    
def two_point_function_fourier(W, lags):
    
    """
    inputs: weight matrix, maximum lag, bin width for cross-correlation
    """    
    
    import numpy as np
    import math
    import params; reload(params)
    from phi import phi_prime
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
    
    phi_r = rates_ss(W)

    wmax = 1. / ((lags[1]-lags[0])) * 2 * math.pi
    dw = wmax/lags.size
    
    w = -wmax/2. + dw*np.arange(lags.size)
#    w[np.where(np.abs(w)<1e-6)] = 1e-6    

#    w = np.arange(-wmax, wmax, dw) * math.pi
#    w, trash = inv_f_trans_james(lags, np.zeros(lags.shape))
    Nw = w.size  
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1 = np.diag(phi_1)   
    W = np.dot(phi_1, W)
    
#    phi_r = np.diag(phi_r)    
    
    C2f = np.zeros((N,N,Nw), dtype=complex) # fourier transform of stationary cross-covariance matrix
    
    for o in range(Nw):
#        C2f[:,:,o] = np.dot(phi_r*Delta[:,:,o], Delta[:,:,o].conj().T)
        F1 = linear_response_fun(w[o], W)
        F2 = linear_response_fun(w[o], W)
        C2f[:,:,o] = np.dot(phi_r*F1, F2.conj().T)
#        
    return C2f, w

def two_point_function(W, lags):
    
    import numpy as np
    import params; reload(params)
    
    par = params.params()
    N = par.N
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')    
    
    C2f, w = two_point_function_fourier(W, lags)

    C2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(C2f, axes=2), axis=2), axes=2) / (lags[1]-lags[0])
    for i in range(N):
        for j in range(N):
            C2[i,j,:] -= C2[i,j,0]

    C2 = C2.real

    return C2
    
def two_point_function_fourier_pop(W, W0, lags, ind):
        
    import numpy as np
    import math
    import params; reload(params)
    from phi import phi_prime
    
    par = params.params()
    N = par.N   
    gain = par.gain
    b = par.b
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
    
    phi_r = rates_ss(W)

    wmax = 1. / ((lags[1]-lags[0])) * 2 * math.pi
    dw = wmax/lags.size    
    w = -wmax/2. + dw*np.arange(lags.size)
    Nw = w.size  
        
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)
    phi_1 = np.diag(phi_1)   
    W = np.dot(phi_1, W)
        
    C2f = np.zeros((Nw), dtype=complex) # fourier transform of stationary cross-covariance matrix       
    
    W0ind = W0[ind,:]
    W0ind = W0ind[:,ind]
    
    for o in range(Nw):
        F1 = linear_response_fun(w[o], W)
        F1lambda = np.dot(F1[ind][:], np.diag(phi_r))
        C2f[o] = np.sum( W0ind*np.dot(F1[ind][:], F1lambda.T.conj()), axis=(0,1)) / ind.size**2
#        
    return C2f, w


def two_point_function_pop(W, W0, lags, ind_pop):
    
    import numpy as np
    
    C2f, w = two_point_function_fourier_pop(W, W0, lags, ind_pop)

    C2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(C2f))) / (lags[1]-lags[0])
    C2 -= C2[0]

    C2 = C2.real

    return C2    
    


def three_point_function(W, lags1, lags2):
    
    import numpy as np
    import params; reload(params)
    
    par = params.params()
    N = par.N
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
        
    C3f, w3 = three_point_function_fourier(W, lags1, lags2)
    Nw3 = w3.size       
    Nt3 = Nw3    
    ind0 = Nt3/2

    C3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(C3f, axes=(3,4)), axes=(3,4)) , axes=(3,4)) / (lags1[1]-lags1[0])
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C3[i,j,k,:,ind0] -= C3[i,j,k,0,ind0] 
                C3[i,j,k,ind0,:] -= C3[i,j,k,ind0,0]
    C3 = C3.real
    
    return C3
    

def three_point_function_fourier(W, lags1, lags2):
    '''
    corresponds to three_point_function_fourier_Yu in Structure_Driven_Activity
    '''
    
    import numpy as np
    import math
    import params; reload(params)
    from phi import phi_prime
    
    from tensor_prods import tensor_aaa_d
    from tensor_prods import tensor_M_T    
    
    par = params.params()
    b = par.b
    gain = par.gain
    N = par.N
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
        
    phi_r = rates_ss(W)    
    print phi_r    
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = np.diag(phi_prime(g0,gain))    

    if lags1.size != lags2.size:
        raise Exception('Two lag arrays should be same size')

    wmax = 1. / ((lags1[1]-lags1[0])) * 2 * math.pi
    Nw = lags1.size
    dw = wmax/Nw    
    w_array = -wmax/2. + dw*np.arange(Nw)
    
#    w_array[np.where(np.abs(w_array)<1e-6)] = 1e-6    
    
    C3f = np.zeros((N,N,N,Nw,Nw), dtype='complex128')
    I = np.eye(N)    
    
    W = np.dot(phi_1, W)
    phi_r = np.dot(np.linalg.inv(I-W), phi_r)

    
    for o1 in range(Nw):
        for o2 in range(Nw):
            w1 = w_array[o1]
            w2 = w_array[o2]
            
            F1 = linear_response_fun(w1, W)
            F2 = linear_response_fun(w2, W)
            F12 = linear_response_fun(-w1-w2, W)
            Fb1 = np.dot(F1, g_fun(w1)*W)
            Fb2 = np.dot(F2, g_fun(w2)*W)
            Fb12 = np.dot(F12, g_fun(-w1-w2)*W)
            
            C3f[:,:,:,o1,o2] = tensor_aaa_d(F12,F1,F2,phi_r) 
            C3f[:,:,:,o1,o2] += tensor_M_T(F12, tensor_aaa_d(Fb12.conj().T, F1, F2, phi_r))
            C3f[:,:,:,o1,o2] += np.transpose(tensor_M_T(F1, tensor_aaa_d(Fb1.conj().T, F2, F12, phi_r)), (2,0,1))
            C3f[:,:,:,o1,o2] += np.transpose(tensor_M_T(F2, tensor_aaa_d(Fb2.conj().T, F12, F1, phi_r)), (1,2,0))
    
    return C3f, w_array


def three_point_function_pop(W, W0, lags1, lags2, ind_pop):
    
    import numpy as np
    import params; reload(params)
    
    par = params.params()
    N = par.N
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
        
    C3f, w3 = three_point_function_fourier_pop(W, W0, lags1, lags2, ind_pop)
    Nw3 = w3.size       
    Nt3 = Nw3    
    ind0 = Nt3/2

    C3 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(C3f, axes=(0,1)), axes=(0,1)) , axes=(0,1)) / (lags1[1]-lags1[0])
    C3[:,ind0] -= C3[0,ind0] 
    C3[ind0,:] -= C3[ind0,0]
    C3 = C3.real.T
    
    return C3


def three_point_function_fourier_pop(W, W0, lags1, lags2, ind):
    '''
    population-averaged (i,j,i) three-point function
    '''
    
    import numpy as np
    import math
    import params; reload(params)
    from phi import phi_prime
    from inv_f_trans_james_old import inv_f_trans_james_2d    
    
    from tensor_prods import tensor_aaa_d
    from tensor_prods import tensor_M_T    
    
    par = params.params()
    b = par.b
    gain = par.gain
    N = par.N
    
    if N != W.shape[0]:
        raise Exception('Mismatch between system size and weight matrix dimensions')
        
    phi_r = rates_ss(W) 
    print phi_r    
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = phi_prime(g0,gain)  

    if lags1.size != lags2.size:
        raise Exception('Two lag arrays should be same size')

    wmax = 1. / ((lags1[1]-lags1[0])) * 2 * math.pi
    Nw = lags1.size
    dw = wmax/Nw    
    w_array = -wmax/2. + dw*np.arange(Nw)
    
#    w_array[np.where(np.abs(w_array)<1e-6)] = 1e-6    
    
    C3f = np.zeros((Nw,Nw), dtype='complex128')
    I = np.eye(N)    
    
    W = np.dot(np.diag(phi_1), W)
    phi_r = np.dot(np.linalg.inv(I-W), phi_r)

    W0ind = W0[ind,:]
    W0ind = W0ind[:,ind]
    
    for o1 in range(Nw):
        for o2 in range(Nw):
            w1 = w_array[o1]
            w2 = w_array[o2]
            
            F1 = linear_response_fun(w1, W)
            F2 = linear_response_fun(w2, W)
            F12 = linear_response_fun(-w1-w2, W)
            Fb1 = np.dot(F1, g_fun(w1)*W)
            Fb2 = np.dot(F2, g_fun(w2)*W)
            Fb12 = np.dot(F12, g_fun(-w1-w2)*W)

#            F1pop = np.sum(F12*F1, axis=0) / ind.size
#            F2pop = np.sum(F2, axis=0) / ind.size
#            F12_F1pop = np.sum(F12*F1*F2, axis=0) / ind.size**2                    
#            C3f[o1,o2] += np.dot(phi_r_ind*F1pop, F2pop) - np.dot(F12_F1pop, phi_r_ind) # Poisson term

#            F12_F1pop = np.sum(F12*F1, axis=0) / ind.size
#            F2pop = np.sum(F2, axis=0) / ind.size           
#            F12_F1_F2pop0 = np.outer(np.sum(F12*F1*F2, axis=0), np.ones(ind.size)) / ind.size**2                    
#            C3f[o1,o2] += np.dot(np.dot(phi_1*F12_F1pop, Fb2.conj()), F2pop*phi_r_ind) #- np.sum(phi_1*F12_F1_F2pop0*Fb2.conj()*phi_r_ind, axis=(0,1))
#
#            F12_F1pop = np.outer(np.sum(F12, axis=0), np.sum(F1, axis=0)) / ind.size
#            F2pop = np.sum(F2, axis=0) / ind.size
#            F12_F2_F1pop0 = np.outer(np.sum(F12*F2, axis=0), np.sum(F1, axis=0)) / ind.size**2                
#            C3f[o1,o2] += np.sum(np.diag(phi_1*F2pop)*F12_F1pop * Fb1.conj() * np.diag(phi_r_ind), axis=(0,1)) #-  np.sum(np.diag(phi_r_ind)*F12_F2_F1pop0*np.diag(phi_1)*Fb1.conj(), axis=(0,1))              
#            
#            
#            F12_F1pop = np.outer(np.sum(F1, axis=0), np.sum(F12, axis=0)) / ind.size
#            F2pop = np.sum(F2, axis=0) / ind.size
#            F12_F1_F2pop0 = np.outer(np.sum(F1*F2, axis=0), np.sum(F12, axis=0)) / ind.size**2
#                        
#            C3f[o1,o2] += np.sum(np.diag(phi_1*F2pop)*F12_F1pop*Fb12.conj()*np.diag(phi_r_ind), axis=(0,1)) #- np.sum(np.diag(phi_1)*F12_F1_F2pop0*np.diag(phi_r_ind)*Fb12.conj(), axis=(0,1))
#
            C3f[o1,o2] += np.sum(W0ind * np.dot(F12[ind]*F1[ind], np.transpose(np.dot(F2[ind], np.diag(phi_r)))), axis=(0,1)) / ind.size**2
            C3f[o1,o2] += np.sum(W0ind * np.dot(np.dot( np.dot(F12[ind]*F1[ind], np.diag(phi_1)), Fb2.conj()), np.transpose(np.dot(F2[ind], np.diag(phi_r)))), axis=(0,1)) / ind.size**2
            C3f[o1,o2] += np.sum(W0ind * np.dot(F12[ind] * np.transpose(np.dot(Fb1.conj(), np.transpose(np.dot(F1[ind], np.diag(phi_r))))), np.transpose(np.dot(F2[ind], np.diag(phi_1)))), axis=(0,1)) / ind.size**2
            C3f[o1,o2] += np.sum(W0ind * np.dot(F1[ind] * np.transpose(np.dot(Fb12.conj(), np.transpose(np.dot(F12[ind], np.diag(phi_r))))), np.transpose(np.dot(F2[ind], np.diag(phi_1)))), axis=(0,1)) / ind.size**2
#            
#    
    return C3f, w_array


def three_point_function_fourier_resum(W, lags1, lags2):
    '''
    adapted from Yu Hu's matlab code
    NEEDS ADAPTING FOR NONLINEARITY (compare above three point function w yu's three point function
    '''
    
    import numpy as np
    import params; reload(params)
    from phi import phi_prime
    from inv_f_trans_james_old import inv_f_trans_james_2d    
    
    from tensor_prods import tensor_aaa_d
    from tensor_prods import tensor_M_T    
    
    par = params.params()
    b = par.b
    gain = par.gain
    N = par.N
    
    phi_r = rates_ss(W)    
    
    g0 = np.dot(W,phi_r) + b
    phi_1 = np.diag(phi_prime(g0,gain))    

    w_array1, w_array2, trash = inv_f_trans_james_2d(lags1, lags2, np.zeros((lags1.size, lags2.size)))
    Nw = w_array1.size
    if w_array1.size != w_array2.size:
        raise Exception('two lag arrays should be same size')

def g_fun(w):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    tau = par.tau
    
    taud = 0.
    
#    g = tau / (1j*w*tau + 1)
    g = np.exp(-1j*w*taud) / ((1+1j*w*tau)**2)

    return g
    
def linear_response_fun(w, W):
    
    import params; reload(params)
    import numpy as np    
    
    par = params.params()
    N = par.N

    Gamma = g_fun(w)*W
    Delta = np.linalg.inv(np.eye(N) - Gamma)
    
    return Delta
    
def linear_response_fun_pop(w, W, ind):
    
    import numpy as np
    Delta = linear_response_fun(w, W)

    Delta = Delta[ind][ind]

    Delta_pop = np.sum(Delta, axis=0) / ind.size
    Delta_pop2 = np.sum(Delta*Delta.conj(), axis=0) / ind.size**2
    
    return Delta_pop, Delta_pop2

def dWdt(W0, W, r0, C2, C2_lagdiff, C3, L, Q, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    Nt = t_th.size    
    N = par.N
    Ne = par.Ne
    WmaxE = par.WmaxE    
    dt_th = t_th[1]-t_th[0]

    dWdt = np.zeros((N,N))   
    
    for ii in range(Ne):
        for jj in range(Ne):

            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[jj]*(-par.A2minus)*par.tauminus*par.eta
            dWdt[ii,jj] += W0[ii,jj]*np.sum(C2[ii,jj,:]*L[:])*dt_th
            
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.tauy*par.eta
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*np.sum( (C2[ii,jj,:])*np.sum(Q, axis=1)*dt_th)*dt_th
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*np.sum(Q*C2_lagdiff[ii,jj,:,:], axis=(0,1))*dt_th*dt_th
            dWdt[ii,jj] += W0[ii,jj]*r0[jj]*np.sum( C2[ii,ii,:]*np.sum(Q[:,:], axis=0)*dt_th)*dt_th
            dWdt[ii,jj] += W0[ii,jj]*np.sum(C3[ii,jj,ii,:,:]*Q, axis=(0,1))*dt_th*dt_th
#
            dWdt[ii,jj] -= W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.eta
#            dWdt[ii,jj] -= W0[ii,jj]*2*r0[ii]*np.sum( (C2[ii,jj,:])*Q[:,ind0])*dt_th
#            dWdt[ii,jj] -= W0[ii,jj]*r0[jj]*np.sum( C2[ii,ii,ind0]*Q[ind0,ind0])*dt_th
#            dWdt[ii,jj] -= W0[ii,jj]*np.sum(C3[ii,jj,ii,:,ind0]*Q[:,ind0])*dt_th


            if W[ii,jj]+dWdt[ii,jj]*dt < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
            if W[ii,jj]+dWdt[ii,jj]*dt > WmaxE:  
                dWdt[ii,jj] = (WmaxE - W[ii,jj])/dt
                
    return dWdt
    
def dWdt_rates(W0, W, r0, L, Q, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    N = par.N
    Ne = par.Ne
    WmaxE = par.WmaxE    

    dWdt = np.zeros((N,N))          
    
    for ii in range(Ne):
        for jj in range(Ne):
            
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[jj]*(-par.A2minus)*par.tauminus*par.eta
            
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.tauy*par.eta
            dWdt[ii,jj] -= W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.eta

            if W[ii,jj]+dWdt[ii,jj] < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
            if W[ii,jj]+dWdt[ii,jj] > WmaxE:  
                dWdt[ii,jj] = (WmaxE - W[ii,jj])/dt
    return dWdt

def dWdt_pairwise(W0, W, r0, C2, C2_lagdiff, L, Q, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    Nt = t_th.size    
    N = par.N
    Ne = par.Ne
    WmaxE = par.WmaxE    
    dt_th = t_th[1]-t_th[0]

    dWdt = np.zeros((N,N))   
    ind0 = Nt/2        
    
    for ii in range(Ne):
        for jj in range(Ne):
            
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[jj]*(-par.A2minus)*par.tauminus*par.eta
            dWdt[ii,jj] += W0[ii,jj]*np.sum(C2[ii,jj,:]*L[:])*dt_th
            
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.tauy*par.eta
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*np.sum( (C2[ii,jj,:])*np.sum(Q, axis=1)*dt_th)*dt_th
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*np.sum(Q*C2_lagdiff[ii,jj,:,:], axis=(0,1))*dt_th*dt_th
            dWdt[ii,jj] += W0[ii,jj]*r0[jj]*np.sum( C2[ii,ii,:]*np.sum(Q[:,:], axis=0)*dt_th)*dt_th

            dWdt[ii,jj] -= W0[ii,jj]*r0[ii]*r0[ii]*r0[jj]*par.A3plus*par.tauplus*par.eta
            dWdt[ii,jj] -= W0[ii,jj]*2*r0[ii]*np.sum( (C2[ii,jj,:])*Q[:,ind0])*dt_th
            dWdt[ii,jj] -= W0[ii,jj]*r0[jj]*np.sum( C2[ii,ii,ind0]*Q[ind0,ind0])*dt_th

            if W[ii,jj]+dWdt[ii,jj] < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
            if W[ii,jj]+dWdt[ii,jj] > WmaxE:  
                dWdt[ii,jj] = (WmaxE - W[ii,jj])/dt
    return dWdt
    
def dpdt(p, p0, r0, C2pop, C2pop_auto, C2pop_lagdiff, C3pop, L, Q, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    Nt = t_th.size    
    ind0 = Nt/2    
    dt_th = t_th[1]-t_th[0]    

    dpdt = 0
    
    if p > 0:
        dpdt += p0*r0**2*(-par.A2minus)*par.tauminus*par.eta
        dpdt += np.sum(C2pop*L)*dt_th
    
    if p < par.WmaxE*p0:
        dpdt += p0*r0**3*par.A3plus*par.tauplus*par.tauy*par.eta
        dpdt += r0*np.sum( C2pop*np.sum(Q, axis=1)*dt_th)*dt_th
        dpdt += r0*np.sum(Q*C2pop_lagdiff, axis=(0,1))*dt_th*dt_th
        dpdt += r0*np.sum( C2pop_auto*np.sum(Q, axis=0)*dt_th)*dt_th
        dpdt += np.sum(C3pop*Q, axis=(0,1))*dt_th*dt_th
        
        dpdt -= p0*r0**3*par.A3plus*par.tauplus*par.eta
        dpdt -= 2*r0*np.sum(C2pop*Q[:,ind0])*dt_th
        dpdt -= r0*np.sum(C2pop_auto[ind0]*Q[ind0,ind0])*dt_th
        dpdt -= np.sum(C3pop[:,ind0]*Q[:ind0])*dt_th
                    
    return dpdt
    
    
def dpdt_pairwise(p, p0, r0, C2pop, C2pop_auto, C2pop_lagdiff, L, Q, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    Nt = t_th.size    
    ind0 = Nt/2   
    dt_th = t_th[1]-t_th[0]     

    dpdt = 0
    
    if p > 0:
        dpdt += p0*r0**2*(-par.A2minus)*par.tauminus*par.eta
        dpdt += np.sum(C2pop*L)*dt_th
    
    if p < par.WmaxE*p0:
        dpdt += p0*r0**3*par.A3plus*par.tauplus*par.tauy*par.eta
        dpdt += r0*np.sum( C2pop*np.sum(Q, axis=1)*dt_th)*dt_th
        dpdt += r0*np.sum(Q*C2pop_lagdiff, axis=(0,1))*dt_th*dt_th
        dpdt += r0*np.sum( C2pop_auto*np.sum(Q, axis=0)*dt_th)*dt_th
        
        dpdt -= p0*r0**3*par.A3plus*par.tauplus*par.eta
        dpdt -= 2*r0*np.sum(C2pop*Q[:,ind0])*dt_th
        dpdt -= r0*np.sum(C2pop_auto[ind0]*Q[ind0,ind0])*dt_th
        
    return dpdt
    
def dpdt_rates(p, p0, r0, t_th, dt):
    
    import params; reload(params)
    par = params.params()

    dpdt = 0
    
    if p > 0:
        dpdt += p0*r0**2*(-par.A2minus)*par.tauminus*par.eta
    
    if p < par.WmaxE*p0:
        dpdt += p0*r0**3*par.A3plus*par.tauplus*par.tauy*par.eta
        dpdt -= p0*r0**3*par.A3plus*par.tauplus*par.eta

    return dpdt
    
def dWdt_pairSTDP(W0, W, r0, C2, L, t_th, dt):
    
    import numpy as np
    import params; reload(params)
    par = params.params()
    
    N = par.N
    Ne = par.Ne
    WmaxE = par.WmaxE    
    dt_th = t_th[1]-t_th[0]
    
    dWdt = np.zeros((N,N))   
    
    print par.A2minus    
    
    for ii in range(Ne):
        for jj in range(Ne):
#            dWdt[ii,jj] += W0[ii,jj]*np.sum(r0[ii]*r0[jj]*L)*dt_th
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[jj]*(-par.A2minus)*par.tauminus*par.eta
            dWdt[ii,jj] += W0[ii,jj]*r0[ii]*r0[jj]*(par.A2minus*2)*par.tauplus*par.eta
            dWdt[ii,jj] += W0[ii,jj]*np.sum(C2[ii,jj,:]*L)*dt_th
    

            if W[ii,jj]+dWdt[ii,jj]*dt < 0:
                dWdt[ii,jj] = -W[ii,jj]/dt
            if W[ii,jj]+dWdt[ii,jj]*dt > WmaxE:  
                dWdt[ii,jj] = (WmaxE - W[ii,jj])/dt
                
    return dWdt