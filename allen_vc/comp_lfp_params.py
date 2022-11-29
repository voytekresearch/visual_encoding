"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""
# FOOOF is causing some annoying warnings about ragged arrays
import warnings
warnings.filterwarnings("ignore")

# imports
import os
import numpy as np
import pandas as pd
from time import time as timer
from time import ctime as time_now
from utils import hour_min_sec
from fooof import FOOOFGroup

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_IN = "data/lfp_psd/natural_movie" # folder containing output of epoch_lfp.py
RELATIVE_PATH_OUT = "data/lfp_params/natural_movie" # where to save output relative to both paths above

# settings - analysis details
N_JOBS = 8 # number of jobs to run in parallel for psd_array_multitaper()
PEAK_WIDTH_LIMITS = [2, 20] # default: (0.5, 12.0))
MAX_N_PEAKS = 4 # (default: inf)
MIN_PEAK_HEIGHT = 0 # (default: 0)
PEAK_THRESHOLD =  2 # (default: 2)
AP_MODE = 'knee'

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # id files of interst and loop through them
    files = os.listdir(f'{MANIFEST_PATH}/{RELATIVE_PATH_IN}')
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}: \t{time_now()}")
        print(f"    {fname_in}")

        # load LFP epochs
        data_in = np.load(f"{MANIFEST_PATH}/{RELATIVE_PATH_IN}/{fname_in}")

        # parameterize 
        df = spec_param_3d(data_in['psd'], data_in['freq'])
        
        # save results 
        for base_path in [PROJECT_PATH, MANIFEST_PATH]:
            dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
            fname_out = fname_in.replace('_psd.npz', f'_params.pkl')
            pd.to_pickle(f"{dir_results}/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def spec_param_3d(psd, freq):
    for i_chan in range(len(psd)):
        # parameterize
        params = FOOOFGroup(peak_width_limits = PEAK_WIDTH_LIMITS,
                        max_n_peaks = MAX_N_PEAKS,
                        min_peak_height = MIN_PEAK_HEIGHT,
                        peak_threshold=PEAK_THRESHOLD,
                        aperiodic_mode=AP_MODE, verbose=False)
        params.fit(freq, psd[i_chan])

        # convert results to df
        df_i = params_to_df(params, MAX_N_PEAKS)
        df_i['chan_idx'] = i_chan
        df_i['epoch_idx'] = np.arange(len(df_i))

        # aggregate across channels
        if i_chan == 0:
            df = df_i
        else:
            df = pd.concat([df, df_i], axis=0)

    return df


def params_to_df(parmas, max_peaks):
    # get per params
    df_per = pd.DataFrame(parmas.get_params('gaussian'),
        columns=['cf','pw','bw','idx'])

    # get ap parmas
    df_ap = pd.DataFrame(parmas.get_params('aperiodic'),  
        columns=['offset', 'knee', 'exponent'])

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