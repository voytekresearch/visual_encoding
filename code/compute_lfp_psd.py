"""
Compute PSD for LFP epochs. Analyzes input of allen_vc.epoch_lfp.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_more_repeats' # name of input/output folders (stimulus of interest)

# imports
import os
import numpy as np
from time import time as timer
import neo
from mne.time_frequency import psd_array_multitaper

# imports - custom
import sys
sys.path.append("allen_vc")
from utils import hour_min_sec
from neo_utils import get_analogsignal
print('Imports complete...')

# settings - analysis details
N_JOBS = 8 # number of jobs to run in parallel for psd_array_multitaper()
F_RANGE = [1, 300]

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    dir_results = f'{PROJECT_PATH}/data/lfp_data/lfp_psd/{STIM_CODE}'
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # id files of interst and loop through them
    dir_input = f'{PROJECT_PATH}/data/blocks/lfp/{STIM_CODE}'
    files = os.listdir(dir_input)
    for i_file, fname_in in enumerate(files):
        
        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        # load block and extract lfp
        block = neo.io.NeoMatlabIO(f"{dir_input}/{fname_in}").read_block()
        lfp = get_analogsignal(block, 'lfp')

        # compute psd
        psd, freq = psd_array_multitaper(lfp, FS, fmin=F_RANGE[0], 
                                         fmax=F_RANGE[1], n_jobs=N_JOBS)

        # save results
        fname_out = fname_in.replace('.mat', '.npz')
        np.savez(f"{dir_results}/{fname_out}", psd=psd, freq=freq) 

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()