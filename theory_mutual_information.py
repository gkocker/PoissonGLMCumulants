'''
Calculate mutual information between activity and a stimulus, assuming gaussian conditional distribution of activity
'''

import numpy as np
import sympy as sp
from scipy.stats import multivariate_normal

def multivar_gaussian_logprob(x, mu, sig):

    N = mu.shape[0]

    part1 = -.5 * (x-mu).dot(np.linalg.inv(sig)).dot(x-mu)

    # sig = sig.real
    u, v = np.linalg.eig(sig)

    part2 = (N/2.) * np.log(2*np.pi) + .5 * np.log(np.sum(u))
    return part1 - part2


def multivar_gaussian_sum_logprob(x, mu1, mu2, sig1, sig2):

    N = x.shape[0]
    p1 = multivar_gaussian_logprob(x, mu1, sig1)

    u1, v1 = np.linalg.eig(sig1.real)
    u2, v2 = np.linalg.eig(sig2.real)
    Z1 = np.sum(u1)
    Z2 = np.sum(u2)
    x_mu1 = (x-mu1).dot(np.linalg.inv(sig1)).dot(x-mu1)
    x_mu2 = (x-mu2).dot(np.linalg.inv(sig2)).dot(x-mu2)

    p2 = np.log(1 + Z2 / Z1 * np.exp(x_mu1 - x_mu2))
    p2b = np.log1p(Z2 / Z1 * np.exp(x_mu1 - x_mu2))
    if np.abs(p2 - p2b) > 1e-5:
        raise Exception('p(r|s1) / p(r|s2) too large - check stimulus labeling')

    return p1 + p2


def mutual_inf_split(mu1, mu2, sig1, sig2 , p_stim1=0.5, p_stim2=0.5):
    '''
    mutual information with estimating p(mixture) to second order in log likelihood with splitting of density into 4 components
    :param mu1:
    :param mu2:
    :param sig1:
    :param sig2:
    :param p_stim1:
    :param p_stim2:
    :return:
    '''

    H_act_stim = p_stim1*entropy_gaussian(mu1, sig1) + p_stim2*entropy_gaussian(mu2, sig2)  # conditional entropy of activity given stimulus

    H0 = entropy_gaussian_mixture_0(mu1, mu2, sig1, sig2, p_stim1, p_stim2)

    weight_split, mu_split, sig_split = split_mixture(mu1, mu2, sig1, sig2, p_stim1, p_stim2)
    # H0 = 0
    H2 = 0
    for n in range(len(weight_split)):
        # H2 += weight_split[n]/2. * np.sum(F_fun_split(mu_split[n], weight_split, mu_split, sig_split)*sig_split[n], axis=(0, 1))
        H2 += weight_split[n]/2. * np.sum(F_fun(mu_split[n], mu1, mu2, sig1, sig2,p_stim1, p_stim2) * sig_split[n], axis=(0, 1))
        # H0 -= weight_split[n]*np.log(multivar_gaussian_prob(mu_split[n], mu_split[n], sig_split[n]))

    H_act = H0 - H2

    return H_act - H_act_stim


def mutual_inf(mu1, mu2, sig1, sig2 , p_stim1=0.5, p_stim2=0.5):
    '''

    :param mu1:
    :param mu2:
    :param sig1:
    :param sig2:
    :param p_stim1:
    :param p_stim2:
    :return:
    '''

    H_act_stim = p_stim1*entropy_gaussian(mu1, sig1) + p_stim2*entropy_gaussian(mu2, sig2)  # conditional entropy of activity given stimulus
    H_act = entropy_gaussian_mixture_2(mu1, mu2, sig1, sig2, p_stim1, p_stim2)

    return H_act - H_act_stim


def entropy_gaussian(mu, sig):
    return multivariate_normal.entropy(mu, sig)

def entropy_gaussian_mixture_0(mu1, mu2, sig1, sig2, weight1, weight2):
    '''
    estimate entropy of a gaussian mixture with two components by expanding logarithm of each component to 0th order
    using taylor expansion of log as in Huber et al., IEEE MFI 2008, Appendix A
    :param mu: mean vector
    :param sig: covariance matrix
    :return: shannon entropy
    '''

    H0 = -weight1*np.log(multivar_gaussian_prob(mu1, mu1, sig1)) - weight2*np.log(multivar_gaussian_prob(mu2, mu2, sig2))

    return H0


def entropy_gaussian_mixture_2(mu1, mu2, sig1, sig2, weight1, weight2):
    '''
    estimate entropy of a gaussian mixture with two components by expanding logarithm of each component to 2nd order
    using taylor expansion of log as in Huber et al., IEEE MFI 2008
    :param mu: mean vector
    :param sig: covariance matrix
    :return: shannon entropy
    '''

    H0 = entropy_gaussian_mixture_0(mu1, mu2, sig1, sig2, weight1, weight2)

    H2 = weight1/2.*np.sum(F_fun(mu1, mu1, mu2, sig1, sig2, weight1, weight2)*sig1, axis=(0,1))
    H2 += weight2/2.*np.sum(F_fun(mu2, mu1, mu2, sig1, sig2, weight1, weight2)*sig2, axis=(0,1))

    return H0 - H2


def multivar_gaussian_prob(x, mu, sig):
    return multivariate_normal.pdf(x, mu, sig)


def F_fun(x, mu1, mu2, sig1, sig2, weight1, weight2):

    N = mu1.shape[0]
    f = mixture_prob(x, mu1, mu2, sig1, sig2, weight1, weight2)
    sig1inv = np.linalg.inv(sig1)
    sig2inv = np.linalg.inv(sig2)

    F = weight1*np.dot(sig1inv, (1./f * np.dot((x-mu1).reshape(N,1), Del_mixture(x, mu1, mu2, sig1, sig2, weight1, weight2).reshape(1,N)) + np.dot((x-mu1).reshape(N,1), np.dot(sig1inv, (x-mu1).reshape(N,1)).T)-np.eye(N)))*multivar_gaussian_prob(x, mu1, sig1)
    F += weight2*np.dot(sig2inv, (1./f * np.dot((x-mu2).reshape(N,1), Del_mixture(x, mu1, mu2, sig1, sig2, weight1, weight2).reshape(1,N)) + np.dot((x-mu2).reshape(N,1), np.dot(sig2inv, (x-mu2).reshape(N,1)).T)-np.eye(N)))*multivar_gaussian_prob(x, mu2, sig2)
    F /= f

    return F


def split_mixture(mu1, mu2, sig1, sig2, weight1, weight2):
    ''' parameters for splitting 1-d standard normal (Huber et al., On Entropy Approximation for Gaussian Mixtures, 2008)'''
    Nsplit = 4
    sig_tilde = 0.51751260421
    weight_tilde = [0.12738084098, 0.37261915901, 0.37261915901, 0.12738084098]
    mu_tilde = [-1.4131205233, -0.44973059608, 0.44973059608, 1.4131205233]

    ''' split first component '''
    u, v = np.linalg.eig(sig1)  # eigenvalue decomposition
    u = np.real(u).astype('float64')
    v = np.real(v).astype('float64')
    d = np.where(u == np.amax(u))[0]
    D = np.diag(u)
    D[d, d] *= sig_tilde ** 2

    mu_split = []
    sig_split = []
    weight_split = []

    for n in range(Nsplit):
        mu = 1 * mu1
        mu[d] += u[d] ** .5 * mu_tilde[n]
        mu_split.append(mu)

        sig = v.dot(D).dot(v.T)
        sig_split.append(sig)

        weight_split.append(weight_tilde[n] * weight1)

    ''' split second component '''
    u, v = np.linalg.eig(sig2)  # eigenvalue decomposition
    u = np.real(u).astype('float64')
    v = np.real(v).astype('float64')
    d = np.where(u == np.amax(u))[0]
    D = np.diag(u)
    D[d, d] *= sig_tilde ** 2

    for n in range(Nsplit):
        mu = 1 * mu2
        mu[d] += u[d] ** .5 * mu_tilde[n]
        mu_split.append(mu)

        sig = v.dot(D).dot(v.T)
        sig_split.append(sig)

        weight_split.append(weight_tilde[n] * weight2)

    return weight_split, mu_split, sig_split


def F_fun_split(x, weight_split, mu_split, sig_split, mu1, mu2, sig1, sig2, weight1, weight2):

    N = mu_split[0].shape[0]

    ''' now have new mixture with Nsplit * 2 components, compute second-order term '''
    # f = mixture_prob_split(x, mu_split, sig_split, weight_split)
    f = mixture_prob(x, mu1, mu2, sig1, sig2, weight1, weight2)
    F = np.zeros((N, N))

    for n in range(len(mu_split)):
        mu = mu_split[n]
        sig = sig_split[n]
        siginv = np.linalg.inv(sig)

        # F += np.dot(siginv, (1./f * np.dot((x-mu).reshape(N,1), Del_mixture_split(x, mu_split, sig_split, weight_split).reshape(1,N)) + np.dot((x-mu).reshape(N,1), np.dot(siginv, (x-mu).reshape(N,1)).T)-np.eye(N)))*multivar_gaussian_prob(x, mu, sig)
        F += np.dot(siginv, (1./f * np.dot((x-mu).reshape(N,1), Del_mixture(x, mu1, mu2, sig1, sig2, weight1, weight2).reshape(1,N)) + np.dot((x-mu).reshape(N,1), np.dot(siginv, (x-mu).reshape(N,1)).T)-np.eye(N)))*multivar_gaussian_prob(x, mu, sig)

    return F


def mixture_prob(x, mu1, mu2, sig1, sig2, weight1, weight2):
    return weight1*multivar_gaussian_prob(x, mu1, sig1) + weight2*multivar_gaussian_prob(x, mu2, sig2)


def mixture_prob_split(x, mu_split, sig_split, weight_split):
    Nsplit = len(mu_split)
    p = 0.
    for n in range(Nsplit):
        p += weight_split[n]*multivar_gaussian_prob(x, mu_split[n], sig_split[n])

    return p

def Del_gaussian(x, mu, sig):
    return np.dot(np.linalg.inv(sig), (x-mu))*multivar_gaussian_prob(x, mu, sig)


def Del_mixture(x, mu1, mu2, sig1, sig2, weight1, weight2):
    return weight1*Del_gaussian(x, mu1, sig1) + weight2*Del_gaussian(x, mu2, sig2)


def Del_mixture_split(x, mu_split, sig_split, weight_split):
    Nsplit = len(mu_split)
    Del = weight_split[0]*Del_gaussian(x, mu_split[0], sig_split[0])
    for n in range(1, Nsplit):
        Del += weight_split[n]*Del_gaussian(x, mu_split[n], sig_split[n])

    return Del


if __name__ == 'mean':

    N = 2
    weight_vec = [.5, .5]

    mu_list = [np.array([1., -.5]), np.array([1., 1.])]
    sig_list = [np.diag([.5, .5]), np.diag([.5, .5])]

    Inf = mutual_inf(mu_list[0], mu_list[0], sig_list[0], sig_list[0], .5, .5)
    Inf_split = mutual_inf_split(mu_list[0], mu_list[0], sig_list[0], sig_list[0], .5, .5)

    print 'No splitting, I=', Inf
    print 'With splitting, I=', Inf_split