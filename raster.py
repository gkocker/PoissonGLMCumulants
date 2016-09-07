# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:58:58 2015

@author: gabeo

make raster plot, having already generated list of spike times and indices in spktimes, with length numspikes
"""

import matplotlib.pyplot as plt

def raster(spktimes, numspikes, tstop, savefile = None, size = None):
    """
    raster plot
    """

    rast = plt.figure(figsize=size)

    for i in range(0,numspikes):
        plt.plot(spktimes[i, 0], spktimes[i, 1], 'k.', markersize=1)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.xticks((0, tstop))
    # plt.xlim((0, max(spktimes[:,0])))
    plt.ylim((0, max(spktimes[:,1])))
    # plt.set_yticks((0, 200))
    
    plt.show()


    if savefile != None:
        rast.savefig(savefile)
        plt.show(rast)
        plt.close(rast)