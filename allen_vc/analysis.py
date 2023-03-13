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



# take the median over all channels for each epoch
def channel_medians(lfp_df, col_names):

    #imports
    import pandas as pd
    
    medians = [0]*len(col_names)
    for epoch in lfp_df.get("epoch_idx").unique():
        epoch_df = lfp_df[lfp_df.get("epoch_idx")==epoch]
        medians = np.vstack((medians, epoch_df.median()))

    medians = np.delete(medians, (0), axis=0)
    return pd.DataFrame(data = medians, columns = col_names)#.drop(columns = 'chan_idx')


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


def sensitivity_analysis(metric, region, psd_file_path, spikes_file_path, running=None, plot_sessions=True,
                         show_sessions=False, mask_nonsig=False, dir_figures=None):

    """
    Performs a sensitivity analysis between PSD values in 30 frequency ranges
    and a given spike metric.

    Parameters
    ----------
    metric : str
        Metric to correlate with PSD values.
    region : str
        Region of the brain to analyze.
    psd_file_path : str
        File path containing PSD data.
    spikes_file_path : str
        File path containing spike data.
    running : bool, optional
        Whether to filter the analysis by running behavior.
    plot_sessions : bool, optional
        Whether to plot the data for each session.
    show_sessions : bool, optional
        Whether to show the data for each session.
    mask_nonsig : bool, optional
        Whether to mask non-significant results in the heat map.
    dir_figures : str, optional
        Directory to save figures.

    Returns
    -------
    None
    """
    
    movie_synchrony_df = pd.read_csv(spikes_file_path)
    
    # filter by behavior
    if running is not None:
        movie_synchrony_df = movie_synchrony_df[movie_synchrony_df.get("running")==running]
    
    # filter by region
    region_movie_synchrony_df = movie_synchrony_df[movie_synchrony_df.get('brain_structure')==region]
    
    # initialize matrix to hold all session data and frequency ranges for heat map
    all_sessions_r, all_sessions_p = np.empty((30, 30, 0)), np.empty((30, 30, 0))
    upper_lims, lower_lims = range(300, 0, -10), range(0, 300, 10)

    # loop through all sessions/files
    for file in os.listdir(psd_file_path):
        
        print(f"Analyzing File:\t{file}")

        # skip weird session with abnormal trial times/counts
        ses_id = int(file.split('_')[0])
        if ses_id == 793224716 or ses_id not in region_movie_synchrony_df.get('session').unique():
            continue

        # load psd data
        data_in = np.load(f'{psd_file_path}/{file}')
        psd, freq = data_in['psd'], data_in['freq']

        # filter by session and get metric
        ses_df = region_movie_synchrony_df[region_movie_synchrony_df.get('session')==ses_id]
        spike_stats = ses_df.get(metric)
        
        if spike_stats is None or len(spike_stats)<2:
            print(f"Not enough data!t # points: {len(spike_stats)}")
            continue
            
        # filter trials based on behavior
        if running is not None:
            trial_filt =  list(ses_df.get("epoch_idx"))
        else:
            trial_filt = None
        
        # use log transformation before average PSD values over range
        freq_range_mat = avg_psd_over_freq_ranges(freq, psd, lower_lims, upper_lims, 
                                                  trial_filter=trial_filt, log_transform=True)

        # compute pearson R between average PSDs and spike metric values for each trial
        r_mat, p_mat = create_r_matrix(spike_stats, freq_range_mat)
        sig_mat = p_mat < 0.05 # determine significance
        
        # set fname_out if saving figures
        if not dir_figures is None:
            fname_out = f'{dir_figures}/{str(ses_id)}_{region}_{metric}.png'
        else:
            fname_out = None

        # plot heat map of r values
        if plot_sessions:
            ses_title = f'{str(ses_id)} - {region} {metric}'
            # apply mask at user discretion
            if mask_nonsig:
                plot_sa_heat_map(r_mat, lower_lims, upper_lims, ses_title,
                             sig_mask=sig_mat, fname_out=fname_out, show_fig=show_sessions)
            else:
                plot_sa_heat_map(r_mat, lower_lims, upper_lims, ses_title,
                                 fname_out=fname_out, show_fig=show_sessions)

        # aggregate all sessions
        all_sessions_r = np.dstack((all_sessions_r, r_mat))
    
    # plot/save session average
    ses_avg_r_mat = np.mean(all_sessions_r, axis=2)
    avg_title = f'Session Average - {region} {metric}'
    if dir_figures:
        plot_sa_heat_map(ses_avg_r_mat, lower_lims, upper_lims, avg_title,
                         fname_out=f'{dir_figures}/session_avg_{region}_{metric}.png')
    else:
        plot_sa_heat_map(ses_avg_r_mat, lower_lims, upper_lims, title)
        
    # save out all_sessions matrix based on pre-defined MATRICES_OUT path
    MATRICES_OUT = f'G:/Shared drives/visual_encoding/data/lfp_data/lfp_r_matrices'
    np.save(f"{MATRICES_OUT}/{psd_file_path.split('/')[-1]}/{region}_{metric}_running={running}", all_sessions_r)

