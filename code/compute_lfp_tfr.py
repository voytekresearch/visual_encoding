"""
Compute PSD for LFP epochs. Analyzes input of allen_vc.epoch_lfp.py.

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_more_repeats' # name of input/output folders (stimulus of interest)

# settings - analysis details
N_JOBS = -1 # number of jobs to run in parallel
F_MIN = 2 # min freq for TF decomposition
F_MAX = 200 # max freq for TF decomposition
N_FREQS = 128 # number of freqs for TF decomposition
DECIM = 25 # decimation factor for TF decomposition
OUTPUT = 'power' # controls values and dimensions of array output by mne.tfr_array_multitaper

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
from tfr_utils import compute_tfr
print('Imports complete...')

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
        print(f"\nAnalyzing {fname_in} ({i_file+1}/{len(files)})")

        # load block and extract lfp
        block = neo.io.NeoMatlabIO(f"{dir_input}/{fname_in}").read_block()
        lfp, time = get_analogsignal(block, 'lfp', return_numpy=True)

        # create mne epochs array and compute power using multitapers
        if output in ['complex', 'phase', 'power']:
            tfr = []
            for i_chan in range(lfp.shape[1]):
                tfr.append(compute_tfr(lfp[:,[i_chan],:], sfreq=FS, f_min=F_MIN, f_max=F_MAX, 
                                          n_freqs=N_FREQS, decim=DECIM, 
                                          n_jobs=N_JOBS, verbose=False))
            tfr = np.swapaxes(tfr,0,1)

        else:
            tfr = compute_tfr(lfp, sfreq=FS, f_min=F_MIN, f_max=F_MAX, 
                                          n_freqs=N_FREQS, decim=DECIM, 
                                          n_jobs=N_JOBS, verbose=False)
        
        # save results
        fname_out = fname_in.replace('.mat', '.npy')
        np.savez(f"{dir_results}/{fname_out}", tfr) 

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()