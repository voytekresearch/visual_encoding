"""
Compute PSD for LFP epochs. Analyzes input of allen_vc.epoch_lfp.py.

"""

# imports
import os
import numpy as np
import pandas as pd
from time import time as timer
from mne.time_frequency import psd_array_multitaper

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_IN = "data/lfp_epochs/natural_movie" # folder containing output of epoch_lfp.py
RELATIVE_PATH_OUT = "data/lfp_psd/natural_movie" # where to save output relative to both paths above

# settings - analysis details
N_JOBS = 8 # number of jobs to run in parallel for psd_array_multitaper()

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
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        # load LFP epochs
        data_in = np.load(f"{MANIFEST_PATH}/{RELATIVE_PATH_IN}/{fname_in}")

        # compute psd
        psd, freq = psd_array_multitaper(data_in['lfp'], FS, n_jobs=N_JOBS)

        # save results
        fname_out = fname_in.replace('.npz', '_psd.npz')
        for base_path in [PROJECT_PATH, MANIFEST_PATH]:
            dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
            np.savez(f"{dir_results}/{fname_out}", psd=psd, freq=freq) 

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def hour_min_sec(duration):
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs


if __name__ == '__main__':
    main()