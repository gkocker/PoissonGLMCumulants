import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if sys.platform == 'darwin':
    save_dir = '/Users/gabeo/Documents/projects/structure_driven_activity/decoding_Ne=200/'
elif sys.platform == 'linux2':
    save_dir = '/home/gabeo/Documents/projects/structure_driven_activity/decoding_Ne=200/'

mu = np.arange(-.1, .1, .001)
ind = mu.shape[0]/2

f_linear = 1.*mu
f_linear[:ind] = 0.

f_quad = mu**2
f_quad[:ind] = 0

plt.figure()
plt.plot(mu, f_linear, 'k', linewidth=2)
savefile = os.path.join(save_dir, 'f-I_linear.pdf')
plt.savefig(savefile)
plt.close()

plt.figure()
plt.plot(mu, f_quad, 'k', linewidth=2)
savefile = os.path.join(save_dir, 'f-I_quadratic.pdf')
plt.savefig(savefile)
plt.close()