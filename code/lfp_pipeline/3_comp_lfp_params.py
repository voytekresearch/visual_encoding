"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie' # name of input/output folders (stimulus of interest)

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
from utils import hour_min_sec

# settings - analysis details
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
    dir_input = f"{PROJECT_PATH}/data/lfp_data/lfp_psd/{STIM_CODE}"
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
    print(f"    File contains {psd.shape[0]} channels and {psd.shape[1]} epochs")

    for i_chan in range(len(psd)):
        # drop trials containing NaNs
        nan_trials = np.isnan(psd[i_chan]).any(axis=1)
        psd_i = psd[i_chan, ~nan_trials]
        if sum(nan_trials) > 0:
            print(f"    Channel {i_chan} has {sum(nan_trials)} trials containing NaNs")

        # parameterize
        params = FOOOFGroup(**SPEC_PARAM_SETTINGS)
        params.fit(freq, psd_i, n_jobs=N_JOBS)

        # convert results to df
        df_i = params_to_df(params, SPEC_PARAM_SETTINGS['max_n_peaks'])
        df_i['chan_idx'] = i_chan
        df_i['epoch_idx'] = np.arange(len(df_i))

        # restore NaN trials
        df_r = pd.DataFrame(np.nan, index=np.arange(len(nan_trials)), columns=df_i.columns)
        df_r.loc[~nan_trials] = df_i

        # aggregate across channels
        if i_chan == 0:
            df = df_r.copy()
        else:
            df = pd.concat([df, df_r], axis=0)

    return df


def params_to_df(params, max_peaks):
    # get per params
    df_per = pd.DataFrame(params.get_params('gaussian'),
        columns=['cf','pw','bw','idx'])

    # get ap parmas
    df_ap = pd.DataFrame(params.get_params('aperiodic'),  
        columns=['offset', 'knee', 'exponent'])

    # get quality metrics
    df_ap['r_squared'] = params.get_params('r_squared')

    # initiate combined df
    df = df_ap.copy()
    columns = []
    for ii in range(max_peaks):
        columns.append([f'cf_{ii}',f'pw_{ii}',f'bw_{ii}'])
    df_init = pd.DataFrame(columns=np.ravel(columns))
    df = df.join(df_init)

    # app gaussian params for each peak fouond
    for i_row in range(len(df)):
        # check if row had peaks
        if df.index[ii] in df_per['idx']:
            # get peak info for row
            df_ii = df_per.loc[df_per['idx']==i_row].reset_index()
            # loop through peaks
            for i_peak in range(len(df_ii)):
                # add peak info to df
                for var_str in ['cf','pw','bw']:
                    df.at[i_row, f'{var_str}_{i_peak}'] = df_ii.at[i_peak, var_str]
    
    return df


if __name__ == '__main__':
    main()