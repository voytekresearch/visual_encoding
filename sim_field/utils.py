"""Utility functions"""

# Imports
import numpy as np
from scipy.optimize import minimize

# Settings

# Functions

def timescale_knee(knee, exponent):
    """
    calculate knee frequency and timecale from FOOOF parameters. This 
    method is detailed in Gao, 2020.

    Parameters
    ----------
    knee : float
        FOOOF aperiodic knee parameter.
    exponent : float
        FOOOF aperiodic exponent parameter..

    Returns
    -------
    tau : float
        timescale (ms).
    knee_hz : float
        knee frequency (Hz).

    """
    # compute knee freq
    knee_hz = knee**(1./exponent)
    
    # compute timescale
    tau = 1./(2*np.pi*knee_hz)
    
    return knee_hz, tau


def autocorr(x, maxlag):
    """ 
    compute autocorrelation
    soure: https://mark-kramer.github.io/Case-Studies-Python/08.html#autocorrelations

    Parameters
    ----------
    x : float
        signal / data for which to compute autocorreltation
    maxlag : int
        number of lags to compute

    Returns
    -------
    xcorr : float
        autocorrelation of x.

    """
    
    xcorr = np.correlate(x - x.mean(), x - x.mean(), 'full')  # Compute the autocorrelation
    xcorr = xcorr / xcorr.max()                               # Convert to correlation coefficients
    xcorr = xcorr[int(xcorr.size//2-maxlag-1) : int(xcorr.size//2+maxlag+1)] # Return only requested lags
    lags = np.linspace(-maxlag, maxlag, len(xcorr))
    
    return xcorr, lags


# fit time constant
def model_acorr(lags, alpha, tau_c):
    """
    model autocorrelation function as exponential decay

    Parameters
    ----------
    lags : float
        array of lag times.
    alpha : float
        amplitude (max correlation).
    tau_c : float
        timescale (decay rate of function).

    Returns
    -------
    model : float
        modelled autocorrelation function.

    """
    model = alpha * np.exp(-np.abs(lags)/(tau_c))
    
    return model

def calc_model_error(tau_c, alpha, empirical, lags):
    """
    calculate model error. Compute the summed absolute differenec between
    the modeled and empirical autocorrelatin function.

    Parameters
    ----------
    tau_c : float
        model parameter - timescale (decay rate of function).
    alpha : float
        model parameter - amplitude (max correlation).
    empirical : float
        autocorrelation function for which to calculate error.
    lags : float
        array of lag times (corresponding to empirical acorr function).

    Returns
    -------
    error : TYPE
        DESCRIPTION.

    """
    
    # comput model
    model = model_acorr(lags, alpha, tau_c)
    
    # calc error
    error = np.sum(np.abs(empirical - model))
    
    return error

def comp_time_constant(signal, maxlag, x0=1):
    """
    compute timescale as decay rate of autocorrelation function

    Parameters
    ----------
    signal : float
        signal / data for which to compute autocorreltation.
    maxlag : int
        number of lags to compute.
    x0 : float, optional
        intial guess for timescale. The default is 1.

    Returns
    -------
    tau_c : float
        timescale fit to signal.

    """
    # compute autocorrelation
    xcorr, lags = autocorr(signal, maxlag)

    # solve for time constant
    result = minimize(calc_model_error, x0=x0, args=(1, xcorr, lags))
    tau_c = result['x']
    
    return tau_c


