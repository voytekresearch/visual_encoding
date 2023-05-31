"""
Parametereize PSDs for LFP epochs. Analyzes output of allen_vc.comp_lfp_psd.py.

"""

# imports
import os
import numpy as np
import pandas as pd
from time import time as timer
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
            os.makedirs(f"{dir_results}/fooof_reports")
    
    # id files of interst and loop through them
    files = os.listdir(f'{MANIFEST_PATH}/{RELATIVE_PATH_IN}')
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        # load LFP epochs
        data_in = np.load(f"{MANIFEST_PATH}/{RELATIVE_PATH_IN}/{fname_in}")

        # parameterize (fit both with and without knee parametere)
        for ap_mode in ['fixed', 'knee']:
            fg = FOOOFGroup(peak_width_limits = PEAK_WIDTH_LIMITS,
                            max_n_peaks = MAX_N_PEAKS,
                            min_peak_height = MIN_PEAK_HEIGHT,
                            peak_threshold=PEAK_THRESHOLD,
                            aperiodic_mode=ap_mode, verbose=False)
            fg.fit(data_in['freq'], data_in['psd'])
            
            # save results 
            for base_path in [PROJECT_PATH, MANIFEST_PATH]:
                dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
                fname_out = fname_in.replace('_psd', f'_params_{ap_mode}')
                fg.save(f"{dir_results}/{fname_out}", save_results=True, 
                        save_settings=True)
                fg.save_report(f"{dir_results}/fooof_reports/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")



if __name__ == '__main__':
    main()