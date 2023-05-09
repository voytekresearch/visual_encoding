# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:49:55 2022

@author: micha

functions for computing spiking statistics

"""

import numpy as np

def comp_fano_factor(spikes, time, bin_duration): 
    import matplotlib.pyplot as plt
    
    # compute spike times
    spike_times = time[np.where(spikes)]
    
    # bin data
    n_bins = int(np.floor((time[-1]-time[0]) / bin_duration))
    spikes_binned, bins, _ = plt.hist(spike_times, n_bins, visible=False);
    plt.close('all')
    
    # compute FF
    fano_factor = np.var(spikes_binned) / np.mean(spikes_binned)
    
    return fano_factor
    
def comp_cov(spikes, time):
    # compute spike times
    spike_times = time[np.where(spikes)]
    
    # compute inter-spike-intervals
    isi = np.diff(spike_times)
    
    # compute CoV
    cov = np.std(isi) / np.mean(isi)
    
    return cov