"""Utility functions for timescale analysis"""

# Imports
import numpy as np
from scipy.optimize import minimize

from neurodsp.spectral import compute_spectrum
from fooof import FOOOF

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

def comp_tau_acorr(signal, fs, maxlag, x0=1):
    """
    compute timescale as decay rate of autocorrelation function

    Parameters
    ----------
    signal : float
        signal / data for which to compute autocorreltation.
    fs : float
        sampling frequency.
    maxlag : int
        number of lags to compute.
    x0 : float, optional
        intial guess for timescale. The default is 1.

    Returns
    -------
    tau_c : float
        timescale of signal.

    """
    # compute autocorrelation
    xcorr, lags = autocorr(signal, maxlag)

    # solve for time constant
    result = minimize(calc_model_error, x0=x0, args=(1, xcorr, lags))
    tau_c = result['x'] / fs
    
    return tau_c

def comp_tau_fooof(signal, fs, peak_width_limits=[2, 20], f_range=[2,200]):
    """
    

    Parameters
    ----------
    signal : float
        signal / data for which to compute autocorreltation.
    fs : float
        sampling freqeuncy.
    peak_width_limits : float, optional
        FOOOF setting - peak width limits. The default is [2, 20].
    f_range : float, optional
        frequency range over which to fit the power spectrum. The default is 
        [2,200].

    Returns
    -------
    tau_c :  float
        timescale of signal.

    """
    # compute psd
    freq, spectrum = compute_spectrum(signal, fs)
    
    # parameterize psd
    sp = FOOOF(peak_width_limits=peak_width_limits, aperiodic_mode='knee')
    sp.fit(freq, spectrum, f_range)
    ap_params = sp.get_params('aperiodic')

    # compute tiemscale from FOOOF parameters
    knee_hz, tau_c = timescale_knee(ap_params[1], ap_params[2])
    
    return tau_c