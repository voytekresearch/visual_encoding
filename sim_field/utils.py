"""Utility functions"""

# Imports
import numpy as np

# Settings

# Functions

def autocorr(x, maxlag):
    """ compute autocorrelation
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