"""
Compute PSD for LFP epochs. Analyzes onput of allen_vc.step3_add_lfp_to_segments.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_more_repeats' # name of input/output folders (stimulus of interest)

# settings - analysis details
N_JOBS = -1 # number of jobs to run in parallel
F_MIN = 2 # min freq for TF decomposition
F_MAX = 200 # max freq for TF decomposition
BANDWIDTH = 1 # frequency bandwidth of the multi-taper window function in Hz

# settings - dataset details
FS = 1250 # LFP sampling freq

# imports
import os
import numpy as np
from time import time as timer
import neo

# imports - custom
import sys
sys.path.append("allen_vc")
from utils import hour_min_sec
from neo_utils import get_analogsignal
from analysis import compute_psd
print('Imports complete...')

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    dir_results = f'{PROJECT_PATH}/data/lfp_data/spectra/psd/{STIM_CODE}'
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # id files of interst and loop through them
    dir_input = f'{PROJECT_PATH}/data/blocks/lfp/{STIM_CODE}'
    files = os.listdir(dir_input)
    for i_file, fname_in in enumerate(files):
        
        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing {fname_in} ({i_file+1}/{len(files)})")

        # load block and extract lfp
        block = neo.io.NeoMatlabIO(f"{dir_input}/{fname_in}").read_block()
        lfp, _ = get_analogsignal(block, 'lfp', return_numpy=True)

        # compute spectral power
        spectra, freq = compute_psd(lfp, FS, fmin=F_MIN, fmax=F_MAX, bandwidth=BANDWIDTH,
                                    n_jobs=N_JOBS, verbose=False)
        
        # save results
        fname_out = fname_in.replace('.mat', '.npz').replace('block', 'spectra')
        np.savez(f"{dir_results}/{fname_out}", spectra=spectra, freq=freq) 

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()