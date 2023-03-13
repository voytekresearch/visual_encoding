import numpy as np
import pandas as pd
import os
from .analysis import avg_psd_over_freq_ranges
from .stats import create_r_matrix
from .plts import plot_sa_heat_map


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
            print(f"Not enough data!\t # points: {len(spike_stats)}")
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