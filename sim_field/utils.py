"""Utility functions for plotting figures, etc"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

##########################################################################
##########################################################################

def plot_coincidences(spikes, fs=1000, maxlags=20, coincidences = None, plot_model=False, normalize = True):
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
    if (n_neurons == 1 or spikes.ndim == 1):
        if coincidences is None:
            coincidences = np.zeros(2 * maxlags + 1)
            coincidences[maxlags] = np.logical_and(spikes[0][1:], spikes[0][:-1]).sum()
            for i in range(1, maxlags + 1):
                coincidences[maxlags + i] = np.logical_and(spikes[0][i:], spikes[0][:-i]).sum()
                coincidences[maxlags - i] = np.logical_and(spikes[0][:-i], spikes[0][i:]).sum()
        plt.bar(range(-maxlags, maxlags + 1), coincidences)
        return
    
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
                if normalize == True:
                    coincidences[i_row, i_col, : ] = coincidences[i_row, i_col, : ] / coincidences[i_row, i_col, : ].max()

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
                # axes[i, j].bar(range(-maxlags, maxlags + 1), coincidences[i+1,j])
                axes[i, j].bar(x_time_ms, coincidences[i+1,j])
                
                # increase text size
                axes[i, j].tick_params(axis='y', labelsize=18)
                axes[i, j].tick_params(axis='x', labelsize=18)

                if plot_model:
                    x0 = [1, 1]
                    lags = range(-maxlags, maxlags + 1)
                    corr = coincidences[i+1,j]
                    result = minimize(calc_model_error, x0=x0, args=(corr, lags))
                    tau_c_fit = result['x'][0]
                    alpha_fit = result['x'][1]
                    model = model_acorr(lags, alpha_fit, tau_c_fit)
                    # cross_covar = alpha * np.exp(-abs(t)/(tau_c * 1000))
                    axes[i, j].plot(lags, model)
                    axes[i, j].set_title('tau: %0.4f' %tau_c_fit)
    # label figure
    fig.text(0.5, -0.05, 'time lag (ms)', ha='center', fontsize=16)
    fig.text(-0.05, 0.5, 'spike coincidences', va='center', fontsize=16, rotation='vertical')
    
    
    return fig, axes
                


def plot_correlations(signals, fs = 1000, plot_model=False, maxlags=20, tau_c=1, alpha=1,
                     plot_all=False):
    """
    plot correlation between multiple signals

    Parameters
    ----------
    signals : 2d array
        matrix of time-series (e.g. firing rates of several neurons)
    fs : float, optional
        sampling frequency. The default is 1000.
    maxlags : int, optional
        maximum lag for correlation calculation. The default is 20.
    tau_c : float, optional
        correlation decay time-constant (param for model). The default is 1.
    alpha : flaot, optional
        maximum correlation (param for model). The default is 1.
    plot_all : bool, optional
        indicate whether to plot the lower triangle only. The default is False.
    plot_model : bool, optional
        indicate whether to plot to correlation model. The default is False.

    Returns
    -------
    fig : pyplot.figure
        figure - correlations between each pair of signals.
    ax : pyplot.axes
        ax - correlations between each pair of signals.

    """
    n_neurons = signals.shape[0]
    if (signals.ndim == 1):
        lags, corr, _, _  = plt.xcorr(signals.astype(float), signals.astype(float), maxlags=maxlags)
        if plot_model:
                # t = np.arange(-maxlags,maxlags)
                x0 = [1, 1]
                result = minimize(calc_model_error, x0=x0, args=(corr, lags))
                tau_c_fit = result['x'][0]
                alpha_fit = result['x'][1]
                model = model_acorr(lags, alpha_fit, tau_c_fit)
                # cross_covar = alpha * np.exp(-abs(t)/(tau_c * 1000))
                plt.plot(lags, model)
                plt.title('tau: %0.4f' %tau_c_fit)
        return
    fig, ax = plt.subplots(nrows=n_neurons, ncols=n_neurons, figsize=(12,10), sharey=True, constrained_layout = True)
    for i_row in range(n_neurons):
        for i_col in range(n_neurons):
            # plot bottom triangle only
            if (not plot_all) & (i_col > i_row): 
                ax[i_row, i_col].axis('off')
                continue
                    
            # plot correlation
            # lags, corr, _, _  = ax[i_row, i_col].xcorr(spikes[i_row].astype(float), spikes[i_col].astype(float), maxlags=maxlags)
            x_1 = signals[i_row].astype(float) - np.mean(signals[i_row].astype(float))
            x_2 = signals[i_col].astype(float) - np.mean(signals[i_col].astype(float))
            lags, corr, _, _ = ax[i_row, i_col].xcorr(x_1, x_2, maxlags=maxlags)
            
            # increase text size
            ax[i_row, i_col].tick_params(axis='y', labelsize=14)
            ax[i_row, i_col].tick_params(axis='x', labelsize=14)
        
            # remove xtick labels from most plots
            if i_row != n_neurons - 1:
                ax[i_row, i_col].set_xticklabels([])
                
            if plot_model:
                # t = np.arange(-maxlags,maxlags)
                x0 = [1, 1]
                result = minimize(calc_model_error, x0=x0, args=(corr, lags))
                tau_c_fit = result['x'][0]
                alpha_fit = result['x'][1]
                model = model_acorr(lags, alpha_fit, tau_c_fit)
                # cross_covar = alpha * np.exp(-abs(t)/(tau_c * 1000))
                ax[i_row, i_col].plot(lags, model)
                ax[i_row, i_col].set_title('tau: %0.4f' %tau_c_fit)
      
                x_bins = np.arange(-maxlags,maxlags)
                x_time_ms = x_bins * (1/fs) *  1000
                cross_covar = alpha * np.exp(-abs(x_bins)/(tau_c * 1000))
                ax[i_row, i_col].plot(x_time_ms, cross_covar)
                
    # label figure
    fig.text(0.5, -0.05, 'time lag (ms)', ha='center', fontsize=18)
    fig.text(-0.05, 0.5, 'correlation', va='center', fontsize=18, rotation='vertical')
    
    return fig, ax

def model_acorr(lags, alpha, tau_c):
    model = alpha * np.exp(-np.abs(lags)/(tau_c))
    
    return model

def calc_model_error(tau_c_alpha, empirical, lags):
    # comput model
    tau_c = tau_c_alpha[0]
    alpha = tau_c_alpha[1]
    model = model_acorr(lags, alpha, tau_c)
    
    # calc error
    error = np.sum(np.abs(empirical - model))
    
    return error

