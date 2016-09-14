import math
import numpy as np
from phi import phi, phi_prime, phi_prime2
import params;

reload(params)


def rates_ss(W):  # set inputs here

    '''
    compute steady-state mean field rates through Euler step
    :param W: weight matrix
    :return: steady-state rates with transfer functions and cellular/synaptic parameters defined in params.py and phi.py
    '''

    par = params.params()
    b = par.b
    gain = par.gain
    tau = par.tau
    N = par.N

    dt = .02 * tau
    Tmax = int(50 * tau / dt)
    a = 1. / tau
    a2 = a ** 2

    r = np.zeros(N)
    s_dummy = np.zeros(N)
    s = np.zeros(N)

    r_vec = np.zeros((N, Tmax))
    for i in range(Tmax):
        s_dummy += dt * (-2 * a * s_dummy - a2 * s) + r * a2 * dt
        s += dt * s_dummy

        g = W.dot(s) + b
        r = phi(g, gain)
        r_vec[:, i] = r

    return r


def rates_1loop(W):
    """
    inputs: weight matrix
    calculate one-loop correction for steady-state firing rates in fluctuation expansion
    """

    import params;
    reload(params)
    from phi import phi_prime
    from phi import phi_prime2

    par = params.params()
    N = par.N
    gain = par.gain
    b = par.b

    phi_r = rates_ss(W)
    Tmax = 100
    dt_ccg = 1.
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    w = np.arange(-wmax, wmax, dw) * math.pi
    dw = dw * math.pi
    Nw = w.size

    g0 = np.dot(W, phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1 = np.diag(phi_1)

    phi_2 = phi_prime2(g0, gain)
    Fbarsum = np.zeros((N, Nw), dtype=complex)

    for o in range(Nw):  # first compute Fbar over dummy frequency
        Fbar1 = np.dot(g_fun(w[o]) * W, linear_response_fun(w[o], np.dot(phi_1, W), phi_r))
        Fbarsum[:, o] = np.dot(Fbar1 * Fbar1.conj(), phi_r)  # sum over first inner vertex

    Fbarsum_int = np.sum(Fbarsum, axis=1) * dw  # integrate over dummy frequency

    F1 = linear_response_fun(0., np.dot(phi_1, W), phi_r)

    r_1loop = np.dot(F1, .5 * phi_2 * Fbarsum_int) / ((2 * math.pi) ** 1)  # sum over second inner vertex
    return r_1loop


def g_fun(w):
    import numpy as np
    import params;
    reload(params)
    par = params.params()
    tau = par.tau

    taud = 0.

    g = np.exp(-1j * w * taud) / ((1 + 1j * w * tau) ** 2)  # alpha function

    return g


def linear_response_fun(w, W, phi_r):
    par = params.params()
    N = par.N

    Gamma = g_fun(w) * W  # W has already been multiplied by the gain of the rate function
    Delta = np.linalg.inv(np.eye(N) - Gamma)

    return Delta


def linear_response_1loop(w, W, phi_r):
    '''
    calculate one-loop correction to the propagator around mean-field theory
    :param w: frequency
    :param W: weight matrix, weighted by postsynaptic gain
    :param phi_r: firing rates
    :return: propagator matrix
    '''

    par = params.params()

    b = par.b
    gain = par.gain
    N = par.N

    Tmax = 100
    dt_ccg = 1
    wmax = 1. / dt_ccg
    dw = 1. / Tmax

    w_calc = np.arange(-wmax, wmax, dw) * math.pi
    dw *= math.pi
    Nw = w_calc.size

    g0 = np.dot(W, phi_r) + b
    phi_1 = phi_prime(g0, gain)
    phi_1_diag = np.diag(phi_1)

    phi_2 = phi_prime2(g0, gain)
    phi_2_diag = np.diag(phi_2)

    F1 = linear_response_fun(w, np.dot(phi_1_diag, W), phi_r)
    Fbar = np.dot(g_fun(w) * W, F1)

    Fbar_int = np.zeros((N, N), dtype='complex128')

    for o in range(Nw):
        Fbar1 = np.dot(g_fun(w_calc[o]) * W, linear_response_fun(w_calc[o], np.dot(phi_1_diag, W), phi_r))
        Fbar2 = np.dot(g_fun(w - w_calc[0]) * W, linear_response_fun(w - w_calc[o], np.dot(phi_1_diag, W), phi_r))
        Fbar_int += np.dot(Fbar1 * Fbar2, np.dot(phi_1_diag, Fbar)) * dw

    linear_response_1loop = np.dot(np.dot(F1, phi_2_diag / 2.), Fbar_int) / (2 * math.pi ** 1)

    return linear_response_1loop
