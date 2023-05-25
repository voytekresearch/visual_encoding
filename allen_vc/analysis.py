# imports
import numpy as np

def compute_tfr(epochs, f_min=None, f_max=None, n_freqs=256, 
                time_window_length=0.5, freq_bandwidth=4, n_jobs=-1, picks=None, 
                average=False, decim=1, verbose=False):
    '''
    This function takes an MNE epochsArray and computes the time-frequency
    representatoin of power using the multitaper method. 
    Due to memory demands, this function should be run on single-channel data, 
    or results can be averaged across trials.
    '''
    # imports
    from mne.time_frequency import tfr_multitaper
    
    # set paramters for TF decomposition
    if f_min is None:
        f_min = (1/(epochs.tmax-epochs.tmin)) # 1/T
    if f_max is None:
        f_max = epochs.info['sfreq'] / 2 # Nyquist

    freq = np.logspace(*np.log10([f_min, f_max]), n_freqs) # log-spaced freq vector
    n_cycles = freq * time_window_length # set n_cycles based on fixed time window length
    time_bandwidth =  time_window_length * freq_bandwidth # must be >= 2

    # TF decomposition using multitapers
    tfr = tfr_multitaper(epochs, freqs=freq, n_cycles=n_cycles, 
                            time_bandwidth=time_bandwidth, return_itc=False, n_jobs=n_jobs,
                            picks=picks, average=average, decim=decim, verbose=verbose)
    
    # extract data
    time = tfr.times
    tfr = tfr.data.squeeze()

    return time, freq, tfr


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

def calculate_spike_metrics(spiketrains):
    """
    calculate spike metrics (mean firing rate, coefficient of variance, 
    SPIKE-distance, SPIKE-synchrony, and correlation coefficient).

    Parameters
    ----------
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object

    Returns
    -------
    mean_firing_rate: float
        mean firing rate over all units during specified epoch.
    unit_firing_rates: list
        list of firing rates for each unit during specified epoch.
    coeff_of_var: float
        coefficient of variation over all units during specified epoch.
    spike_dist: float
        SPIKE-distance (pyspike) over all units during specified epoch.
    spike_sync: float
        SPIKE-synchrony (pyspike) over all units during specified epoch.
    corr_coeff:
        correlation coefficient (elephant) over all units during 
        specified epoch. 
    """
    # Imports
    import elephant
    import quantities as pq
    from neo_utils import gen_pop_spiketrain

    # compute metrics
    unit_firing_rates = [len(spiketrain)/float(spiketrain.duration) \
        for spiketrain in spiketrains]
    mean_firing_rate = sum(unit_firing_rates)/len(spiketrains)
    coeff_of_var = (compute_cv(gen_pop_spiketrain(spiketrains, t_stop=spiketrains[0].t_stop)))
    spike_sync, spike_dist = compute_pyspike_metrics(spiketrains)
    corr_coeff = (elephant.spike_train_correlation.correlation_coefficient(\
        elephant.conversion.BinnedSpikeTrain(spiketrains, bin_size=1 * pq.s)))

    return mean_firing_rate, unit_firing_rates, coeff_of_var, spike_dist, spike_sync, corr_coeff


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

