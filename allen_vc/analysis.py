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


def avg_psd_over_freq_ranges(freq, psd, lower_lims, upper_lims, log_transform=False):
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


def create_r_matrix(statistics, mat):    
    """
    Create a matrix of Pearson correlation coefficients between spike statistics and PSD data.

    Parameters
    ----------
    statistics : array_like
        Array of spike statistics.
    mat : array_like
        3D array of PSD data.

    Returns
    -------
    r_mat : array_like
        2D array of Pearson correlation coefficients.
    """
    # initialize matrix for storage of r values
    r_mat = []
    p_mat = []
    
    # loop through data in mat
    for i, row in enumerate(mat):
        r_mat_row = []
        p_mat_row = []
        for j, col in enumerate(row):
            all_trials = mat[i,j,:]
            
            # disregard invalid ranges
            if all(np.isnan(all_trials)):
                results = [np.nan, np.nan]
            else:
                # some trials have NaN PSD data so filter those out
                results = sts.pearsonr(statistics[~np.isnan(all_trials)], \
                                      all_trials[~np.isnan(all_trials)])
            r_mat_row.append(results[0])
            p_mat_row.append(results[1])
        r_mat.append(r_mat_row)
        p_mat.append(p_mat_row)
        
    return np.array(r_mat), np.array(p_mat)

# combine full pipeline
def sensitivity_analysis(metric, region, psd_file_path, spikes_file_path, plot_sessions=True,
                         show_sessions=False, mask_nonsig=False, dir_figures=None):
    
    movie_synchrony_df = pd.read_csv(spikes_file_path)
    
    # filter for v1 but later do LGd
    region_movie_synchrony_df = movie_synchrony_df[movie_synchrony_df.get('brain_structure')==region]

    # initialize matrix to hold all session data
    all_sessions_r = np.empty((30, 30, 0))
    all_sessions_p = np.empty((30, 30, 0))

    # loop through all sessions/files
    for file in os.listdir(psd_file_path):

        # skip weird session with abnormal trial times/counts
        ses_id = int(file.split('_')[0])
        if ses_id == 793224716 or ses_id not in region_movie_synchrony_df.get('session_id').unique():
            continue

        # log progress
        # print(f"Analyzing File:\t{file}")

        # load data
        data_in = np.load(f'{psd_file_path}/{file}')
        psd, freq = data_in['psd'], data_in['freq']

        # calculate average psd over frequency ranges
        upper_lims = range(300, 0, -10)
        lower_lims = range(0, 300, 10)
        # use log transformation before average PSD values over range
        freq_range_mat = avg_psd_over_freq_ranges(freq, psd, lower_lims, upper_lims, log_transform=True)

        # filter by session and get metric
        ses_id = int(file.split('_')[0])
        ses_df = region_movie_synchrony_df[region_movie_synchrony_df.get('session_id')==ses_id]
        spike_stats = ses_df.get(metric)

        # compute pearson R between average PSDs and spike metric values for each trial
        r_mat, p_mat = create_r_matrix(spike_stats, freq_range_mat)
        sig_mat = p_mat < 0.05 # determine significance
        
        # set fname_out if saving figures
        if not dir_figures is None:
            fname_out = f'{dir_figures}/{region}_{metric}_{str(ses_id)}.png'
        else:
            fname_out = None

        # plot heat map of r values
        if plot_sessions:
            if mask_nonsig:
                plot_sa_heat_map(r_mat, lower_lims, upper_lims, f'{str(ses_id)} - {region} {metric}',
                             sig_mask=sig_mat, fname_out=fname_out, show_fig=show_sessions)
            else:
                plot_sa_heat_map(r_mat, lower_lims, upper_lims, f'{str(ses_id)} - {region} {metric}',
                                 fname_out=fname_out, show_fig=show_sessions)

        # aggregate all sessions
        all_sessions_r = np.dstack((all_sessions_r, r_mat))
    
    # plot session average
    ses_avg_r_mat = np.mean(all_sessions_r, axis=2)
    if dir_figures:
        plot_sa_heat_map(ses_avg_r_mat, lower_lims, upper_lims, f"Session Average - {region} {metric}",
                         fname_out=f'{dir_figures}/{region}_session_avg_{metric}.png')
    else:
        plot_sa_heat_map(ses_avg_r_mat, lower_lims, upper_lims, f"Session Average - {region} {metric}")
        
     # save out all_sessions matrix
    np.save(f"{MATRICES_OUT}/{psd_file_path.split('/')[-1]}/{region}_{metric}", all_sessions_r)

