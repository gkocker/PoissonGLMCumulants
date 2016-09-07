from __future__ import division
import numpy as np
from scipy.linalg import eigvals
import matplotlib.pyplot as plt

def rectifier(x,t=0.0):
    phi = x.copy()
    phi[x<t] = 0.0
    return phi

def rectifier_prime(x,t=0.0):
    phi_prime = np.ones(x.shape)
    phi_prime[x<t] = 0.0
    return phi_prime

class firing_rate_function (object):
    def __init__(self,f,fprime=None):
        self.f = f
        if fprime:  self.prime = fprime
        #deal with case where fprime =None

    def __call__(self, *args, **kwargs):
        return self.f(*args,**kwargs)

rectifier_firing_rate = firing_rate_function(rectifier,rectifier_prime)

class SDA (object):
    def __init__(self,W=None,b=None,phi=None,tau=1.0):
        if phi:
            self.phi=phi
        else:
            self.phi = rectifier_firing_rate

        if W is not None:
            self.W = W
            self.N = W.shape[0]
        else:
            self.W = np.random.random([100,100])
            self.N = 100

        self.tau = tau
        if b is not None:
            self.b = b
        else:
            self.b = np.ones(self.N)/self.tau

    def reset_variables(self):

        del self.steady_state
        del self.g
        del self.gamma
        del self.omega
        del self.prop_fourier
        del self.prop_sum
        del self.two_point_cor
        del self.three_point_cor
        del self.three_point_cor_sum_A
        del self.three_point_cor_sum_B

    def compute_mean_field_steady_state(self,s_0=None,dt=0.01,tol=1e-7):
        # we'll do this with Euler step and check for convergence for now; probably there is a better way
        # in particular, we need a way of finding and selecting from multiple steady states, as in the ring model

        if s_0 is not None:
            s = s_0.copy()
        else:
            s = np.random.random(self.N)

        old_s = s
        delta_s = 1.0 #np.max(np.abs(s - old_s))

        while delta_s > tol:

            arg = np.dot(self.W,old_s) + self.b
            delta_s = (-old_s/self.tau + self.phi(arg))
            s = old_s + dt*delta_s

            delta_s = np.max(np.abs(delta_s))
            old_s = s

        self.steady_state = s

        self.compute_g_steady_state()

        return self.steady_state

    def compute_g_steady_state(self):

        self.g = np.dot(self.W,self.steady_state) + self.b

        return self.g

    def compute_gamma(self):

        #self.gamma = np.zeros(self.W.shape)

        if not hasattr(self,'steady_state'):
            raise Exception('No steady state defined.  Please run compute_mean_field_steady_state()')

        if hasattr(self,'gamma'):
            return self.gamma

        self.gamma = np.eye(self.W.shape[0])

        #g = np.dot(self.W,self.steady_state) + self.b
        self.gamma -=  (self.W.T*self.phi.prime(self.g)).T

        return self.gamma

    def set_omega(self,omega):

        self.omega = omega

    def compute_propagator_fourier(self):  #,omega=None):

        if hasattr(self,'prop_fourier'):
            return self.omega, self.prop_fourier

        self.compute_gamma()

        if not hasattr(self,'omega'):
            self.omega = np.linspace(-10.0,10.0,1000)

        # if omega is not None:
        #     self.omega = omega
        # else:
        #     self.omega = np.linspace(-10.0,10.0,1000)

        prop_shape = (self.W.shape[0],self.W.shape[1],len(self.omega))
        self.prop_fourier = np.empty(prop_shape,dtype='complex128')

        for k, om in enumerate(self.omega):
            delta = -1j*om*np.eye(self.W.shape[0]) + self.gamma
            self.prop_fourier[:,:,k] = np.linalg.inv(delta)   #maybe a better inversion algorithm here

        return self.omega, self.prop_fourier

    def compute_summed_prop(self):

        if hasattr(self,'prop_sum'):
            return self.prop_sum

        self.compute_propagator_fourier()

        self.prop_sum = np.sum(self.prop_fourier,axis=0)/self.N#        Gamma[:,:,i] = np.dot(phi_1,W)*g[i]
#        Delta[:,:,i] = np.linalg.inv((np.eye(N)-Gamma[:,:,i]))

        return self.prop_sum

    def compute_two_point_cor(self):

        self.compute_propagator_fourier()

        # self.two_point_cor = np.zeros([self.N,self.N,len(self.omega)],dtype='complex128')
        # for i in range(self.N):
        #     for j in range(self.N):
        #         for k in range(self.N):
        #             state = self.steady_state[k]
        #             if state<0.0:  state=0.0
        #             self.two_point_cor[i,j,:] += self.prop_fourier[i,k,:]*self.prop_fourier[j,k,:]*state

        #can do this with matrix multiplication
        self.two_point_cor = np.empty([self.N,self.N,len(self.omega)],dtype='complex128')
        for o in range(len(self.omega)):
            self.two_point_cor[:,:,o] = np.dot((self.prop_fourier[:,:,o]*self.phi(self.steady_state)),self.prop_fourier[:,:,o].conj().T)
            # removed a 2pi along with the delta function for frequency

        return self.two_point_cor

    def compute_two_point_cor_population(self):

        self.compute_summed_prop()

        self.two_point_cor_population = np.empty(len(self.omega))

        for o in range(len(self.omega)):
            o2 = M - o
            self.two_point_cor_population[o] = np.dot(self.prop_sum[:,o]*self.phi(self.steady_state), self.prop_sum[:,o2].conj().T)

    def compute_three_point_A_full(self):

        # removed a 2pi along with the delta function for frequency
        # These are going to be BIG BIG arrays!!!  Don't run this in general, only for really small examples (a handful of neurons)

        # not finished, need to fix indices

        self.compute_propagator_fourier()

        phi = self.phi(self.steady_state)

        self.three_point_cor = np.zeros([self.N,self.N,self.N,len(self.omega),len(omega)],dtype='complex128')
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    for r in range(self.N):
                        self.three_point_cor[i,j,] += self.prop_fourier[i,r,:]*self.prop_fourier[j,r,:]*self.prop_fourier[k,r,:]*phi[r]

        return self.three_point_cor

    def compute_three_point_population_A(self):

        self.compute_summed_prop()

        M=len(self.omega)
        m=int(M/2)

        phi = self.phi(self.steady_state)

        self.three_point_cor_sum_A = np.zeros([m,m])

        for m1 in range(m):
            for m2 in range(m):
                for r in range(self.N):
                    o1 = M/4 + m1
                    o2 = M/4 + m2
                    o3 = M - o1 - o2
                    self.three_point_cor_sum_A[o1,o2]+= self.prop_sum[r,o1]*self.prop_sum[r,o2]*self.prop_sum[r,o3]*phi[r]

        return self.three_point_cor_sum_A

    def compute_three_point_population_B(self):

        self.compute_summed_prop()

        M=len(self.omega)
        m = int(M/2)

        phi = self.phi(self.steady_state)
        phi_prime = self.phi.prime(self.steady_state)

        w_sum = np.sum(self.W,axis=0)/float(N)
        F_bar = np.tensordot(w_sum,self.prop_fourier,axes=[[0],[0]])

        self.three_point_cor_sum_B = np.zeros([m,m])

        for m1 in range(m):
            o1 = M/4 + m1
            for m2 in range(m):
                o2 = M/4 + m2
                o3 = M - o1 - o2
                o4 = o1 + o2
                for r in range(self.N):
                    factor = self.prop_sum[r,o1]*self.prop_sum[r,o2]
                    for s in range(self.N):
                        self.three_point_cor_sum_B[o1,o2] += factor*self.prop_sum[s,o3]*F_bar[s,o4]*phi[s]*phi_prime[r]

        return self.three_point_cor_sum_B



""" +++++++++++++++++++++++++++++++++++
MICHAEL'S INSTANTIATION OF CODE FOR 2-CELL NET

def main():
    sda_net = SDA(phi=rectifier_firing_rate)

    # x = np.random.random(10)-0.5
    #
    # print sda_net.phi(x)
    # print sda_net.phi.prime(x)

    #print sda_net.W

    W = 0.5*np.array([[1.,-1.],[-1.,1.]])
    b = 0.5*np.ones(2)

    sda_net = SDA(W=W,b=b)
    #sda_net = SDA()
    s_0 = np.array([0.7,0.1])
    sda_net.compute_mean_field_steady_state(s_0=s_0)
    #sda_net.compute_mean_field_steady_state()
    print sda_net.steady_state
    sda_net.compute_gamma()
    print sda_net.gamma

    # l, v = np.linalg.eig(sda_net.gamma)
    #
    # print l
    # print v

    om, prop = sda_net.compute_propagator_fourier()

    cor = sda_net.compute_two_point_cor()
    #cor_pop = sda_net.compute_two_point_cor_population()

    #directions in which we project population activtiy in computing "summed outputs"
    s_tot = np.array([1.0, 1.0])
    s_diff = np.array([1.0,-1.0])

    cor_tot = np.tensordot(cor,s_tot,axes=([0],[0]))
    cor_tot = np.tensordot(cor_tot,s_tot,axes=([0],[0]))


    cor_diff = np.tensordot(cor,s_diff,axes=([0],[0]))
    cor_diff = np.tensordot(cor_diff,s_diff,axes=([0],[0]))

    print (0.5/np.pi)*np.sum(cor_tot.real)*20/1000.0

    print

+++++++++++++++++++++++++++++++++++"""

""" +++++++++++++++++++++++++++++++++++
Attempt instantiation for random matrix
"""
def main():
    
    sda_net = SDA(phi=rectifier_firing_rate)

#    N=100  #number of cells
##
##    # Building random matices
##    # Dense, random
##    W = np.random.random([N,N])
#
#
#    W = np.random.random([N,N])
#
#
#    #If want, make balanced (or almost, depending on multipler of mean here)
#    balance_level=0  #set to 1 for perfect balance in W itself, 0 to leave alone
#    W = W - balance_level*np.mean(W)  #make 0 mean, sort of like having global inhibition pathway
#    print 'mean of W=' + repr(np.mean(W))
#    
#    
#    #Assuming that slope of phi is 1, rescale W so has spectral radius = spect_rad_desired, less than 1
#    spect_rad_actual=max(abs(eigvals(W)))
#    spect_rad_desired=0.9  
#    W = W*spect_rad_desired/spect_rad_actual
    
    N = 2
    W = np.ones((N,N)) - np.eye(N)
    W[0,1] = .8
    W[1,0] = .1
    
    #additive inputs
    b = 0.5*np.ones(N)
    
    print 'W'    
    print W
    print 'b'    
    print b
    
    sda_net = SDA(W=W,b=b)

    s_0 = np.random.random([N])
  
    sda_net.compute_mean_field_steady_state(s_0=s_0)

    print 'steady state'    
    print sda_net.steady_state
   
    #evaluate at 0 frequency only
#    sda_net.set_omega([0])    
     
    om, prop = sda_net.compute_propagator_fourier()   
    plt.figure(); plt.plot(om,prop[0,1,:])    
    plt.figure(); plt.plot(om,np.imag(prop[0,1,:]))
    
    two_point_corr = sda_net.compute_two_point_cor()
    plt.figure(); plt.plot(om, two_point_corr[0,1,:])
    
    from inv_f_trans import inv_f_trans
    t_th, C2 = inv_f_trans(om, two_point_corr[0,1,:])
    plt.figure(); plt.plot(t_th, C2,'r',linewidth=2)
    three_point_corr_f = sda_net.compute_three_point_population_B()
    three_point_corr = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(three_point_corr_f)))
    ind0 = three_point_corr.shape[0]/2
    plt.figure(); plt.plot(three_point_corr[:,ind0])
    plt.figure(); plt.plot(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(two_point_corr[0,1,:]))))
    
    print 'propagator matrix at 0 freq, should be real'    
    print prop
    
#    #ESB defines summed propsgator as Pbar in notes
#    Pbar=sda_net.compute_summed_prop()
#    
#    print'summed propagator'
#    print Pbar
#    #sda_net.compute_gamma()
#    #print sda_net.gamma
#
#    #element-wise square
#    Pbar2=np.power(Pbar, 2)
#    Pbar3=np.power(Pbar, 3)
#
##    sda_net.compute_g_steady_state()
#    
#    glist=sda_net.g
#    print 'glist' 
#    print glist
#    
#    philist=sda_net.phi(glist)
#    phi_primelist=sda_net.phi.prime(glist)
#
#    print 'philist' 
#    print philist    
#
#    print 'phi_primelist' 
#    print phi_primelist 
#    
#    S2pre=np.multiply(Pbar2,phi_primelist)
#    S2=np.sum(S2pre)
#    S2norm=S2/(N**3)
#
#    print 'S2pre'
#    print S2pre
#    print S2
#
#    S3pre_term1=np.multiply(Pbar3,phi_primelist)
#    S3term1=np.sum(S3pre_term1)
#    
#    prop=np.reshape(prop,(N,N))    
#    print 'prop reshaped'
#    print prop    
#    
#    print  np.shape(prop)
#    print  np.shape(philist)
#   
#    temp=np.dot(prop,philist)
#
#    print  np.shape(temp)
#
#    Pbar3=np.reshape(Pbar3,(N))    
#    print  np.shape(Pbar3)
#    S3term2= 3* np.dot(Pbar3,temp)
#
#    S3=S3term1+S3term2
#    S3norm=S3/(N**3)
#    
#    print 'S3'
#    print S3
#    
#    print '_______________ and we find ... ________'
#    print 'N=' + repr(N) +', Avg 2nd cum=' + repr(S2norm.real) + ',   Avg 3rd cum=' + repr(S3norm.real)
    
if __name__=='__main__':
    main()
