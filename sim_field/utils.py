"""Utility functions for plotting figures, etc"""

import numpy as np
import matplotlib.pyplot as plt

##########################################################################
##########################################################################

def plot_coincidences(spikes, fs=1000, maxlags=20, coincidences = None):
    """
    Count coincidences between spikes trains and plot correlation

    Parameters
    ----------
    spikes : 2d array
        spikes trains
    maxlags : int, optional
        maximum lag for correlation calculation. The default is 20.
    coincidences : 3d array, optional
        spike coincidence (if already computed) The default is None.

    Returns
    -------
    fig : matplotlib.pyplot.figure
        pyplot figure. spike coincidences
    axes : matplotlib.pyplot.axes
        pyplot axes. spike coincidences

    """
    
    n_neurons = spikes.shape[0]
    
    # count coincidence, if neccessary 
    if coincidences is None:
        coincidences = np.zeros((n_neurons, n_neurons, 2 * maxlags + 1))
        for i_row in range(n_neurons):
            for i_col in range(n_neurons):
                # count coincidences for lag=0
                coincidences[i_row, i_col, maxlags] = np.logical_and(spikes[i_row], spikes[i_col]).sum()
                
                # count coincidence for time-lags of interest
                for i in range(1, maxlags + 1):
                    coincidences[i_row, i_col,maxlags + i] = np.logical_and(spikes[i_row][i:], spikes[i_col][:-i]).sum()
                    coincidences[i_row, i_col,maxlags - i] = np.logical_and(spikes[i_row][:-i], spikes[i_col][i:]).sum()

    # create figure
    fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=True, 
                             ncols=n_neurons-1, nrows=n_neurons-1,
                             tight_layout=True)
    for i in range(n_neurons - 1):
        for j in range(n_neurons - 1):
            # plot lower triangle only
            if i<j:
                axes[i, j].axis('off')
            else:
                # convert bins to ms
                x_bins = np.linspace(-maxlags, maxlags, coincidences.shape[2])
                x_time_ms = x_bins * (1/fs) *  1000
                
                # plot coincidences
                axes[i, j].bar(x_time_ms, coincidences[i+1,j])
                
                # increase text size
                axes[i, j].tick_params(axis='y', labelsize=18)
                axes[i, j].tick_params(axis='x', labelsize=18)
                
    # label figure
    fig.text(0.5, -0.05, 'time lag (ms)', ha='center', fontsize=16)
    fig.text(-0.05, 0.5, 'spike coincidences', va='center', fontsize=16, rotation='vertical')
    
    
    return fig, axes

def plot_correlations(signals, fs=1000, maxlags=20, tau_c=1, alpha=1,
                     plot_all=False, plot_model=False):
    
    
    n_neurons = signals.shape[0]
    fig, ax = plt.subplots(nrows=n_neurons, ncols=n_neurons, figsize=(12,10), sharey=True, constrained_layout=True)
    for i_row in range(n_neurons):
        for i_col in range(n_neurons):
            # plot bottom triangle only
            if (not plot_all) & (i_col > i_row): 
                ax[i_row, i_col].axis('off')
                continue
                    
            # plot correlation
            x_1 = signals[i_row].astype(float) - np.mean(signals[i_row].astype(float))
            x_2 = signals[i_col].astype(float) - np.mean(signals[i_col].astype(float))
            ax[i_row, i_col].xcorr(x_1, x_2, maxlags=maxlags)
            
            # increase text size
            ax[i_row, i_col].tick_params(axis='y', labelsize=14)
            ax[i_row, i_col].tick_params(axis='x', labelsize=14)
        
            # remove xtick labels from most plots
            if i_row != n_neurons - 1:
                ax[i_row, i_col].set_xticklabels([])
                
            if plot_model:
                x_bins = np.arange(-maxlags,maxlags)
                x_time_ms = x_bins * (1/fs) *  1000
                cross_covar = alpha * np.exp(-abs(x_bins)/(tau_c * 1000))
                ax[i_row, i_col].plot(x_time_ms, cross_covar)
                
    # label figure
    fig.text(0.5, -0.05, 'time lag (ms)', ha='center', fontsize=18)
    fig.text(-0.05, 0.5, 'correlation', va='center', fontsize=18, rotation='vertical')
    
    return fig, ax
