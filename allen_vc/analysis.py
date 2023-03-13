# imports
import numpy as np

def comp_spike_cov(spike_train):
    """Computes the coefficient of variation (CoV) of the interspike interval (ISI) distribution.

    Parameters
    ----------
    spike_train : neo.SpikeTrain
        Neo SpikeTrain object

    Returns
    -------
    cov : float
        Coefficient of variation (CoV) of the interspike interval (ISI) distribution.
    """
    # account for empty spike_train
    if len(spike_train)==0:
        return 0
    
    # compute interspike intervals
    isi = np.diff(spike_train.times)

    # compute coefficient of variation
    cov = np.std(isi) / np.mean(isi)
    
    # returns as a 'dimensionless string' without float constructor
    return float(cov)


def calculate_spike_metrics(spiketrains):
    """
    calculate spike metrics (mean firing rate, coefficient of variance, 
    SPIKE-distance, SPIKE-synchrony, and correlation coefficient) within
    a specified epoch given a matrix of spike times.

    Parameters
    ----------
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object

    Returns
    -------
    mean_firing_rate: float
        mean firing rate over all units during specified epoch.
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
    #Imports
    import pyspike as spk
    import elephant
    import quantities as pq
    from allen_vc.utils import gen_pop_spiketrain

    # reformat as PySpike object for synchrony analyses
    spk_trains = [spk.SpikeTrain(spiketrain, [spiketrain.t_start, spiketrain.t_stop]) \
        for spiketrain in spiketrains]

    # compute metrics
    unit_firing_rates = [len(spiketrain)/float(spiketrain.duration) \
        for spiketrain in spiketrains]
    mean_firing_rate = sum(unit_firing_rates)/len(spiketrains)
    coeff_of_var = (comp_spike_cov(gen_pop_spiketrain(spiketrains, t_stop=spiketrains[0].t_stop)))
    spike_dist = (spk.spike_distance(spk_trains))
    spike_sync = (spk.spike_sync(spk_trains))
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

