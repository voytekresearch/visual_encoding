"""Utility functions for plotting figures, etc"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

##########################################################################
##########################################################################

def plot_coincidences(spikes, maxlags=20, coincidences = None, plot_model=False):
    n_neurons = spikes.shape[0]
    if (n_neurons == 1):
        if coincidences is None:
            coincidences = np.zeros(2 * maxlags + 1)
            coincidences[maxlags] = np.logical_and(spikes[0][1:], spikes[0][:-1]).sum()
            for i in range(1, maxlags + 1):
                coincidences[maxlags + i] = np.logical_and(spikes[0][i:], spikes[0][:-i]).sum()
                coincidences[maxlags - i] = np.logical_and(spikes[0][:-i], spikes[0][i:]).sum()
        plt.bar(range(-maxlags, maxlags + 1), coincidences)
        return
        
    if coincidences is None:
        coincidences = np.zeros((n_neurons, n_neurons, 2 * maxlags + 1))
        for i_row in range(n_neurons):
            for i_col in range(n_neurons):
                coincidences[i_row, i_col, maxlags] = np.logical_and(spikes[i_row], spikes[i_col]).sum()
                for i in range(1, maxlags + 1):
                    coincidences[i_row, i_col,maxlags + i] = np.logical_and(spikes[i_row][i:], spikes[i_col][:-i]).sum()
                    coincidences[i_row, i_col,maxlags - i] = np.logical_and(spikes[i_row][:-i], spikes[i_col][i:]).sum()

    fig, axes = plt.subplots(figsize=(10, 10), sharex=False, sharey=True, ncols=n_neurons - 1, nrows= n_neurons - 1, tight_layout = True)
    for i in range(n_neurons - 1):
        for j in range(n_neurons - 1):
            if i<j:
                axes[i, j].axis('off')
            else:
                axes[i, j].bar(range(-maxlags, maxlags + 1), coincidences[i+1,j])
                if plot_model:
                    x0 = [1, 1000]
                    lags = range(-maxlags, maxlags + 1)
                    corr = coincidences[i+1,j]
                    result = minimize(calc_model_error, x0=x0, args=(corr, lags), method = 'Nelder-Mead')
                    tau_c_fit = result['x'][0]
                    alpha_fit = result['x'][1]
                    model = model_acorr(lags, alpha_fit, tau_c_fit)
                    # cross_covar = alpha * np.exp(-abs(t)/(tau_c * 1000))
                    axes[i, j].plot(lags, model)
                    axes[i, j].set_title('tau: %0.4f' %tau_c_fit)
    return fig, axes

def plot_correlations(spikes, plot_model=False, maxlags=20, tau_c=1, alpha=1,
                     plot_all=False):
    n_neurons = spikes.shape[0]
    fig, ax = plt.subplots(nrows=n_neurons, ncols=n_neurons, figsize=(14,14), sharey=True, tight_layout = True)
    for i_row in range(n_neurons):
        for i_col in range(n_neurons):
            # plot bottom triangle only
            if (not plot_all) & (i_col > i_row): 
                ax[i_row, i_col].axis('off')
                continue
                    
            # plot correlation
            lags, corr, _, _  = ax[i_row, i_col].xcorr(spikes[i_row].astype(float), spikes[i_col].astype(float), maxlags=maxlags)
            
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

