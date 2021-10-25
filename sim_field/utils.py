"""Utility functions for plotting figures, etc"""

import numpy as np
import matplotlib.pyplot as plt

##########################################################################
##########################################################################

def plot_coincidences(spikes, maxlags=20, coincidences = None):
    n_neurons = spikes.shape[0]
    if coincidences is None:
        coincidences = np.zeros((n_neurons, n_neurons, 2 * maxlags + 1))
        for i_row in range(n_neurons):
            for i_col in range(n_neurons):
                coincidences[i_row, i_col, maxlags] = np.logical_and(spikes[i_row], spikes[i_col]).sum()
                for i in range(1, maxlags + 1):
                    coincidences[i_row, i_col,maxlags + i] = np.logical_and(spikes[i_row][i:], spikes[i_col][:-i]).sum()
                    coincidences[i_row, i_col,maxlags - i] = np.logical_and(spikes[i_row][:-i], spikes[i_col][i:]).sum()

    fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=True, ncols=4, nrows=4)
    for i in range(n_neurons - 1):
        for j in range(n_neurons - 1):
            if i<j:
                axes[i, j].axis('off')
            else:
                axes[i, j].bar(range(-maxlags, maxlags + 1), coincidences[i+1,j])
    return fig, axes

def plot_correlations(spikes, plot_model=False, maxlags=20, tau_c=1, alpha=1,
                     plot_all=False):
    n_neurons = spikes.shape[0]
    fig, ax = plt.subplots(nrows=n_neurons, ncols=n_neurons, figsize=(14,14), sharey=True)
    for i_row in range(n_neurons):
        for i_col in range(n_neurons):
            # plot bottom triangle only
            if (not plot_all) & (i_col > i_row): 
                ax[i_row, i_col].axis('off')
                continue
                    
            # plot correlation
            ax[i_row, i_col].xcorr(spikes[i_row].astype(float), spikes[i_col].astype(float), maxlags=maxlags)
            
            if plot_model:
                t = np.arange(-maxlags,maxlags)
                cross_covar = alpha * np.exp(-abs(t)/(tau_c * 1000))
                ax[i_row, i_col].plot(t, cross_covar)
      
    return fig, ax
