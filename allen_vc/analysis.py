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

