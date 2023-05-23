# -*- coding: utf-8 -*-
"""
Utility functions for TFR analysis
"""

# Imports
import numpy as np


def compute_tfr(epoch_data, sfreq, f_min=None, f_max=None, n_freqs=256, 
                time_window_length=0.5, freq_bandwidth=4, n_jobs=-1, 
                output='power', decim=1, verbose=False):
    '''
    This function takes an array (n_epochs, n_channels, n_times) and computes the time-frequency
    representatoin of power using the multitaper method. 
    Due to memory demands, this function should be run on single-channel data, 
    or results can be averaged across trials.
    
    Parameters
    ----------
    epoch_data : 3D array
        Array of shape (n_epochs, n_channels, n_times) containing the data.
    sfreq : float
        Sampling frequency of the data.
    f_min : float
        Minimum frequency of interest. If None, set to 1/T, where T is the length of the time window.
    f_max : float
        Maximum frequency of interest. If None, set to Nyquist frequency (sfreq/2).
    n_freqs : int
        Number of frequencies to use for the TF decomposition.
    time_window_length : float
        Length of the time window (in seconds) to use for the TF decomposition.
    freq_bandwidth : float
        Bandwidth of the frequency window (in Hz) to use for the TF decomposition.
    n_jobs : int
        Number of jobs to run in parallel. If -1, use all available cores.
    output : str
        Type of output to return. The default is 'power'
    decim : int
        Decimation factor to use for the TF decomposition.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    tfr : 4D array
        Array of shape (n_epochs, n_channels, n_freqs, n_times) containing the TF representation.
    '''
    # imports
    from mne.time_frequency import tfr_array_multitaper
    
    # set paramters for TF decomposition
    if f_min is None:
        T = epoch_data.shape[2]/sfreq
        f_min = (1/T)
    if f_max is None:
        f_max = sfreq / 2 # Nyquist

    freq = np.logspace(*np.log10([f_min, f_max]), n_freqs) # log-spaced freq vector
    n_cycles = freq * time_window_length # set n_cycles based on fixed time window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_array_multitaper(epoch_data, freqs=freq, n_cycles=n_cycles, 
                            time_bandwidth=time_bandwidth, output=output, n_jobs=n_jobs,
                            decim=decim, verbose=verbose)

    # deleted parameters: picks, average, return_itc

    # return full array for now
    return tfr


def zscore_tfr(tfr):
    """
    Normalize time-frequency representation (TFR) by z-scoring each frequency.
    TFR should be 2D (frequency x time).

    Parameters
    ----------
    tfr : 2D array
        Time-frequency representation of power (spectrogram).

    Returns
    -------
    tfr_norm : 2D array
        Z-score normalized TFR.
    """
    
    # initialize 
    tfr_norm = np.zeros(tfr.shape)
    
    # z-score normalize 
    for i_freq in range(tfr.shape[0]):
        tfr_norm[i_freq] = (tfr[i_freq] - np.mean(tfr[i_freq])) / np.std(tfr[i_freq])
        
    return tfr_norm


def subtract_baseline(signals, time, t_baseline):
    """
    Subtract baseline from signals. Baseline is defined as the mean of the
    signal between t_baseline[0] and t_baseline[1]. Signals should be 2D
    (signals x time).

    Parameters
    ----------
    signals : 2D array
        Signals to be baseline corrected.
    time : 1D array
        Time vector.
    t_baseline : 1D array
        Time range for baseline (t_start, t_stop).

    Returns
    -------
    signals_bl : 2D array
        Baseline corrected signals.
    """
    
    # initialize
    signals_bl = np.zeros_like(signals)
    
    # subtract baseline from each signal
    for ii in range(len(signals)):
        mask_bl = ((time>t_baseline[0]) & (time<t_baseline[1]))
        bl = np.mean(signals[ii, mask_bl])
        signals_bl[ii] = signals[ii] - bl
    
    return signals_bl


def crop_tfr(tfr, time, time_range):
    """
    Crop time-frequency representation (TFR) to time_range.
    TFR can be mulitdimensional (time must be last dimension).

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of
        the last dimension of tfr).
    time_range : 1D array
        Time range to crop (t_start, t_stop).

    Returns
    -------
    tfr, time : array, array
        Cropped TFR and time vector.
    """
    
    tfr = tfr[..., (time>time_range[0]) & (time<time_range[1])]
    time = time[(time>time_range[0]) & (time<time_range[1])]
    
    return tfr, time


def downsample_tfr(tfr, time, n):
    """
    Downsample time-frequency representation (TFR) to n time bins.
    TFR can be mulitdimensional (time must be last dimension)

    Parameters
    ----------
    tfr : array
        Time-frequency representation of power (spectrogram).
    time : 1D array
        Associated time vector (length should be equal to that of 
        the last dimension of tfr).
    n : int
        Desired number of time bins after downsampling.

    Returns
    ------- 
    tfr, time : array, array
        Downsampled TFR and time vector.
    """

    # determine step size for downsampling and counnt number of samples
    n_samples = len(time)
    step = int(np.floor(tfr.shape[-1]/n))

    # downsample
    tfr = tfr[..., np.arange(0, n_samples-1, step)] 
    time = time[np.arange(0, n_samples-1, step)] 
    
    return tfr, time


def preprocess_tfr(tfr, time, downsample_n=None, edge=None, average_trials=True, z_score=True, t_baseline=None):

    # downsample
    if not downsample_n is None:
        tfr, time = downsample_tfr(tfr, time, downsample_n)

    # crop edge effects
    if not edge is None:
        tfr, time = crop_tfr(tfr, time, [time[0]+edge/2,time[-1]-edge/2])

    # average spectrogram over trials
    if average_trials:
        tfr = np.median(tfr, axis=0)

    # normalize (zscore)
    if z_score:
        tfr = zscore_tfr(tfr)

    # subtract basline
    if not t_baseline is None:
        tfr = subtract_baseline(tfr, time, t_baseline)

    return tfr, time


def load_tfr_results(fname, preprocess=True, downsample_n=None, edge=None, average_trials=True, z_score=True, t_baseline=None):
    # load data
    data_in = np.load(fname)

    # unpack
    tfr = np.squeeze(data_in['tfr']) # remove extra dim (previously channels dim)
    time = data_in['time']
    freq = data_in['freq']

    # set default basline time
    if t_baseline == 'default':
        t_baseline=[time[0],0]

    # pre-process
    if preprocess:
        tfr, time = preprocess_tfr(tfr, time, downsample_n=downsample_n, edge=edge, 
                                   average_trials=average_trials, z_score=z_score,
                                   t_baseline=t_baseline)
        
    return time, freq, tfr

