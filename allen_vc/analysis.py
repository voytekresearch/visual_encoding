# imports
import numpy as np


def compute_psd(signal, fs, fmin=0, fmax=np.inf, bandwidth=None,
                n_jobs=-1, verbose=False):
    '''
    This function takes an array (n_epochs, n_channels, n_times) and computes the power spectral 
    spectra using the multitaper method. 
    
    Parameters
    ----------
    signal : 3D array
        Array of shape (n_epochs, n_channels, n_times) containing the data.
    fs : float
        Sampling frequency of the data.
    fmin : float
        Minimum frequency of interest. The default is 0, which will result in 1/T.
    fmax : float
        Maximum frequency of interest. The default is np.inf, which will result in the Nyquist frequency.
    bandwidth : float
        Frequency bandwidth of the multi-taper window function in Hz. If None, set to 
        8*(fs / n_times) (see mne.time_frequency.psd_array_multitaper).
    n_jobs : int
        Number of jobs to run in parallel. If -1, use all available cores.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    psd : 3D array
        Array of shape (n_epochs, n_channels, n_freqs) containing the power spectral density.
    freq : 1D array
        Array of shape (n_freqs,) containing the frequencies.
    '''

    # imports
    from mne.time_frequency import psd_array_multitaper
    
    # set paramters for TF decomposition
    if bandwidth is None:
        bandwidth = 8*(fs / signal.shape[2])

    # TF decomposition using multitapers
    spectra, freq = psd_array_multitaper(signal, fs, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
                                         n_jobs=n_jobs, verbose=verbose)

    return spectra, freq


def compute_cv(spiketrain):
    """Compute the coefficient of variation (CV) of the interspike interval (ISI)
     of a spike train.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain
        Neo SpikeTrain object

    Returns
    -------
    cov : float
        Coefficient of variation (CV) of the interspike interval (ISI) distribution.
    """

    # check if there are any spikes
    if len(spiketrain)==0:
        return np.nan
    
    # compute interspike intervals
    isi = np.diff(spiketrain.times)

    # compute coefficient of variation
    cv = np.float(np.std(isi) / np.mean(isi))
    
    return cv


def compute_pyspike_metrics(spiketrains, interval=None):
    """
    compute spike synchrony and spike distance using PySpike.

    Parameters
    ----------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object
    interval : list, optional
        Interval over which to compute synchrony and distance. The default is None.
        If None, the entire duration of the SpikeTrains object is used.

    Returns
    -------
    spike_dist: float
        SPIKE-distance (pyspike) over all units (during specified interval).
    spike_sync: float
        SPIKE-synchrony (pyspike) over all units (during specified interval).
    """

    # imports
    import pyspike as spk
    import quantities as pq

    # reformat Neo objects to PySpike objects
    spk_trains = [spk.SpikeTrain(spiketrain, [spiketrain.t_start, spiketrain.t_stop]) \
        for spiketrain in spiketrains]
    
    # compute metrics
    if interval is None:
        interval = [spiketrains[0].t_start.item(), spiketrains[0].t_stop.item()]
    spike_dist = spk.spike_distance(spk_trains, interval=interval)
    spike_sync = spk.spike_sync(spk_trains, interval=interval)

    return spike_dist, spike_sync


def avg_psd_over_freq_ranges(freq, psd, lower_lims, upper_lims, trial_filter=None, log_transform=False):
    """
    Compute the average power spectral density (PSD) over frequency ranges for a given set of trials.

    Parameters
    ----------
    freq : array_like
        Array of frequencies.
    psd : array_like
        Array of PSD values.
    lower_lims: array_like
        Array of lower limits for frequency ranges.
    upper_lims: array_like
        Array of upper limits for frequency ranges.
    log_transform : bool, optional
        Whether to take the log10 of the PSD values before computing the average. Default is False.

    Returns
    -------
    mat : array_like
        3D array of average PSD values over frequency ranges for each trial.
    """
    # determine number of trials in block
    num_trials = psd.shape[1]
    
    # intialize empty matrix for storage
    mat = np.empty((30, 30, 0))
    
    # loop through trials
    for trial in range(num_trials):

        if trial_filter is not None and trial not in trial_filter:
            continue
        
        # take median across channels
        psd_trial = np.median(psd[:,trial,:], axis=0)
        trial_mat = []
        
        # loop through upper/lower frequency limits
        for upper_lim in upper_lims:
            row = []
            for lower_lim in lower_lims:
                if upper_lim<=lower_lim:
                    row.append([np.nan])
                    continue
                    
                # filter for range and compute average
                psd_range = psd_trial[(freq>lower_lim) & (freq<=upper_lim)]
                if log_transform:
                    psd_avg = np.mean(np.log10(psd_range))
                else:
                    psd_avg = np.mean(psd_range)
                row.append([psd_avg])
            trial_mat.append(row)
            
        # stack matrices for each trial
        mat = np.dstack((mat, trial_mat))
        
    return mat

