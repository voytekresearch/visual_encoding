"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_more_repeats' # name of input/output folders (stimulus of interest)

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
from utils import hour_min_sec

# settings
OVERWRITE = False # whether to overwrite existing results

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
    dir_results = f'{PROJECT_PATH}/data/lfp_data/params/{INPUT_TYPE}/{STIM_CODE}'
    print(f"Saving results to: {dir_results}")
    if not os.path.exists(f"{dir_results}/by_session"):
        os.makedirs(f"{dir_results}/by_session")
    
    # initialize output
    params_list = []

    # id files of interest and loop through them
    dir_input = f"{PROJECT_PATH}/data/lfp_data/spectra/{INPUT_TYPE}/{STIM_CODE}"
    files = os.listdir(dir_input)
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}: \t{time_now()}")
        print(f"    Filename: {fname_in}")

        # check if file already exists
        fname_out = fname_in.replace('.npz', f'.csv').replace('spectra', 'params')
        if not OVERWRITE and os.path.exists(f"{dir_results}/by_session/{fname_out}"):
            print("    File already exists, skipping")
            continue

        # load LFP power spectra
        data_in = np.load(f"{dir_input}/{fname_in}")

        if INPUT_TYPE == 'psd':
            psd, freq = data_in['spectra'], data_in['freq']
            df = spec_param_3d(psd, freq)

        elif INPUT_TYPE == 'tfr':
            tfr, freq = data_in['tfr'], data_in['freq']

            # NOTE: spec_param_3d is computing across all time windows
            #       pd concat is putting all trials/epochs together


            # move axis so that labeling/fitting is congruent with psd
            tfr = np.moveaxis(tfr, 3, 1)
            df = pd.concat([spec_param_3d(tfr[i,:,:,:], freq).assign(window_idx=i) 
                for i in range(tfr.shape[-1])])

            window_indices = df['window_idx'].copy()
            df['window_idx'] = df['epoch_idx']
            df['epoch_idx'] = window_indices

        else:
            raise RuntimeError("INPUT_TYPE must be psd or tfr")

        # aggregate across files
        df['session'] = fname_in.split('_')[1].split('.')[0]
        params_list.append(df)
        
        # save results 
        df.to_csv(f"{dir_results}/by_session/{fname_out}", index=False)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # aggregate across files
    params = pd.concat(params_list, axis=0)
    params.to_csv(f"{dir_results}/lfp_params.csv", index=False)
    
    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def spec_param_3d(psd, freq):

    # display progress
    print(f"    File contains {psd.shape[1]} channels and {psd.shape[0]} epochs")

    # check missingness
    nan_trials = np.isnan(psd).sum(axis=1).sum(axis=1) != 0
    # psd_clean = psd[~nan_trials]
    print(f"    Found {nan_trials.sum()} trials with nan values\n\n")

    # parameterize
    fg = FOOOFGroup(**SPEC_PARAM_SETTINGS)
    fg.set_check_data_mode(False)
    fg._check_freqs = False
    fgs = fit_fooof_3d(fg, freq, psd, n_jobs=N_JOBS)
    df = pd.concat([params.to_df(SPEC_PARAM_SETTINGS['max_n_peaks']) for params in fgs], axis=0)

    # add channel and epoch labels
    n_epochs, n_chans = psd.shape[:2]
    df['chan_idx'] = n_epochs*list(range(n_chans))
    df['epoch_idx'] = np.concatenate([[i]*n_chans for i in range(n_epochs)])

    return df
    

if __name__ == '__main__':
    main()