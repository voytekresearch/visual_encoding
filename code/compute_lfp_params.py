"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'spontaneous/running' # name of input/output folders (stimulus of interest)
BEHAVIOR_LABEL = True # whether or not to include column denoted 'above' or 'below' behavior

# FOOOF is causing some annoying warnings about ragged arrays
import warnings
warnings.filterwarnings("ignore")

# imports - general
import os
import numpy as np
import pandas as pd
from time import time as timer
from time import ctime as time_now
from fooof import FOOOFGroup

# imports - custom
import sys
sys.path.append("allen_vc")
from utils import params_to_df, hour_min_sec

# settings - analysis details
INPUT_TYPE = 'psd' # psd for 3d and tfr for 4d
N_JOBS = -1 # number of jobs for parallel processing, psd_array_multitaper()
SPEC_PARAM_SETTINGS = {
    'peak_width_limits' :   [2, 20], # default: (0.5, 12.0)) - reccomends at least frequency resolution * 2
    'min_peak_height'   :   0, # (default: 0) 
    'max_n_peaks'       :   4, # (default: inf)
    'peak_threshold'    :   2, # (default: 2.0)
    'aperiodic_mode'    :   'knee',
    'verbose'           :   False}

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    dir_results = f'{PROJECT_PATH}/data/lfp_data/lfp_params/{STIM_CODE}'
    if not os.path.exists(f"{dir_results}/by_session"):
        os.makedirs(f"{dir_results}/by_session")
    
    # initialize output
    params_list = []

    # id files of interst and loop through them
    dir_input = f"{PROJECT_PATH}/data/lfp_data/lfp_{INPUT_TYPE}/{STIM_CODE}"
    files = os.listdir(dir_input)
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}: \t{time_now()}")
        print(f"    Filename: {fname_in}")

        # load LFP power spectra
        data_in = np.load(f"{dir_input}/{fname_in}")

        # parameterize PSDs
        df = spec_param_3d(data_in['psd'], data_in['freq'])

        if BEHAVIOR_LABEL:
            df['behavior'] = fname_in.split('_')[-2]

        # aggregate across files
        df['session'] = fname_in.split('_')[0]
        params_list.append(df)
        
        # save results 
        fname_out = fname_in.replace('_psd.npz', f'_params.csv')
        df.to_csv(f"{dir_results}/by_session/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # aggregate across files
    params = pd.concat(params_list, axis=0)
    params.to_csv(f"{dir_results}/lfp_params.csv")
    
    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def spec_param_3d(psd, freq):
    # display progress
    print(f"    File contains {psd.shape[1]} channels and {psd.shape[0]} epochs")

    # loop through trials
    for i_trial in range(len(psd)):
        # drop trials containing NaNs
        nan_chans = np.isnan(psd[i_trial]).any(axis=1)
        psd_i = psd[i_trial, ~nan_chans]
        if sum(nan_chans) > 0:
            print(f"    Trial {i_trial} has {sum(nan_chans)} channels containing NaNs")

        # parameterize
        params = FOOOFGroup(**SPEC_PARAM_SETTINGS)
        params.fit(freq, psd_i, n_jobs=N_JOBS)

        # convert results to df
        df_i = params_to_df(params, SPEC_PARAM_SETTINGS['max_n_peaks'])
        df_i['epoch_idx'] = i_trial
        df_i['chan_idx'] = np.arange(psd.shape[1])[~nan_chans]

        # restore NaN trials
        df_e = pd.DataFrame(np.nan, index=np.arange(psd.shape[1]), columns=df_i.columns)
        df_e.loc[~nan_chans] = df_i

        # aggregate across channels
        if i_trial == 0:
            df = df_e.copy()
        else:
            df = pd.concat([df, df_e], axis=0)

    return df

def spec_param_4d(tfr, freq):
    # display progress
    print(f"    File contains {tfr.shape[2]} time windows, {tfr.shape[1]} channels, and {tfr.shape[0]} epochs")
    # loop through trials
    for i_trial in range(len(tfr)):

    	trial_tfr = np.moveaxis(tfr[i_trial], 2, 0)
        # drop trials containing NaNs
        nan_chans = np.isnan(trial_tfr).any(axis=2)
        tfr_i = trial_tfr[:,:,~nan_chans]
        if sum(nan_chans) > 0:
            print(f"    Trial {i_trial} has {sum(nan_chans)} channels containing NaNs")

        # parameterize
        params = FOOOFGroup(**SPEC_PARAM_SETTINGS)
        fooof_groups = fit_fooof_3d(params, freq, tfr_i, n_jobs=N_JOBS)

        for i_group, group in enumerate(fooof_groups):
	        # convert results to df
	        df_i = params_to_df(group, SPEC_PARAM_SETTINGS['max_n_peaks']) # this should probably be updated
	        df_i['epoch_idx'] = i_trial
	        df_i['chan_idx'] = np.arange(tfr.shape[1])[~nan_chans]

	        # restore NaN trials
	        df_e = pd.DataFrame(np.nan, index=np.arange(tfr.shape[1]), columns=df_i.columns)
	        df_e.loc[~nan_chans] = df_i

	        # aggregate across fooof groups
	        if i_group==0:
	        	df_g = df_e.copy()
	        else:
	        	df = pd.concat([df, df_e], axis=0)

        # aggregate across channels
        if i_trial == 0:
            df = df_g.copy()
        else:
            df = pd.concat([df, df_g], axis=0)

    return df




if __name__ == '__main__':
    main()