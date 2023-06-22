"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_more_repeats' # name of input/output folders (stimulus of interest)
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
from fooof import FOOOFGroup, fit_fooof_3d

# imports - custom
import sys
sys.path.append("allen_vc")
from utils import params_to_df, hour_min_sec

# settings - analysis details
INPUT_TYPE = 'psd' # denoting whether input measures psd or tfr
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
    dir_results = f'{PROJECT_PATH}/data/lfp_data/lfp_params/{STIM_CODE}/{INPUT_TYPE}'
    if not os.path.exists(f"{dir_results}/by_session"):
        os.makedirs(f"{dir_results}/by_session")
    
    # initialize output
    params_list = []

    # id files of interest and loop through them
    dir_input = f"{PROJECT_PATH}/data/lfp_data/lfp_{INPUT_TYPE}/{STIM_CODE}"
    files = os.listdir(dir_input)
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}: \t{time_now()}")
        print(f"    Filename: {fname_in}")

        # load LFP power spectra
        data_in = np.load(f"{dir_input}/{fname_in}")

        if INPUT_TYPE == 'psd':
            psd, freq = data_in['spectra'], data_in['freq']
            df = spec_param_3d(psd, freq)

        elif INPUT_TYPE == 'tfr':
            tfr, freq = data_in['tfr'], data_in['freq']
            df = pd.concat([spec_param_3d(tfr[:,:,:,i], freq).assign(time_window=i) 
                for i in range(tfr.shape[-1])])

        else:
            raise RuntimeError("INPUT_TYPE must be psd or tfr")

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

    # check missingness
    nan_trials = np.isnan(psd).sum(axis=1).sum(axis=1) != 0
    psd_clean = psd[~nan_trials]
    print(f"    Found {nan_trials.sum()} trials with nan values")

    # parameterize
    fg = FOOOFGroup(**SPEC_PARAM_SETTINGS)
    fgs = fit_fooof_3d(fg, freq, psd_clean, n_jobs=N_JOBS)
    df = pd.concat([params.to_df(SPEC_PARAM_SETTINGS['max_n_peaks']) for params in fgs], axis=0)

    # add channel and epoch labels
    n_chans, n_epochs = psd_clean.shape[1], psd_clean.shape[0]
    df['chan_idx'] = n_epochs*list(range(n_chans))
    df['epoch_idx'] = np.concatenate([[i]*n_chans for i in range(n_epochs)])

    return df
    

if __name__ == '__main__':
    main()