"""
Import running velocity time-series, and find running/stationary epochs.
"""

# imports
import os
import numpy as np
import random

# Settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding
RELATIVE_PATH_IN = "data/behavior/running/spontaneous" # where input data is saved relative to both paths above

# settings - dataset details
FS = 2500 # Sampling rate (Hz)
BLOCK = 4

# settings - analysis
SPEED_THRESHOLD = 5 # Any speed greater than this value is considered running (cm/s)
MIN_DURATION = 1 # Minimum time of measured epochs (s)
EPOCH_LENGTH = 10

# Import custom functions			  
import sys
sys.path.append(REPO_PATH)
from allen_vc.epoch_extraction_tools import get_epoch_times, get_random_epoch

def main():
	# Define/create directories for output
	for base_path in [PROJECT_PATH, MANIFEST_PATH]:
		dir_results = f'{base_path}/data/behavior/running/epoch_times'
		if not os.path.exists(dir_results): 
			os.makedirs(dir_results)

	# Initialize storage space
	epoch_times = {}
	random_epochs = {}

    # id files of interest and loop through them
	files = os.listdir(f'{PROJECT_PATH}/{RELATIVE_PATH_IN}')
	files = [f for f in files if f.endswith('.npz')] # .npz files only
	for i_file, fname_in in enumerate(files):
		# displey progress
		session_id = fname_in.split('_')[1].split('.')[0]
		print(f'Analyzing session: \t{session_id}\tBlock: \t{BLOCK}')

		# load running data
		series_data = np.load(f'{PROJECT_PATH}/{RELATIVE_PATH_IN}/{fname_in}')

		# Getting epoch times
		running_epochs, stationary_epochs = \
			get_epoch_times(series_data['velocity'], SPEED_THRESHOLD, MIN_DURATION*FS)
		
		# aggregate results across sessions
		epoch_times[f'{session_id}_running'] = series_data['time'][0] + running_epochs/FS
		epoch_times[f'{session_id}_stationary'] = series_data['time'][0] + stationary_epochs/FS

		# Choose and store random epoch from selection cropped to EPOCH_LENGTH
		random.seed(101)
		random_epochs[f'{session_id}_running'] = get_random_epoch(\
			epoch_times[f'{session_id}_running'], EPOCH_LENGTH)
		random_epochs[f'{session_id}_stationary'] = get_random_epoch(\
			epoch_times[f'{session_id}_stationary'], EPOCH_LENGTH)

	# Save data
	for base_path in [PROJECT_PATH, MANIFEST_PATH]:
		dir_results = f'{base_path}/data/behavior/running/epoch_times'
		fname_out = f'{dir_results}/{RELATIVE_PATH_IN.split("/")[-1]}_{BLOCK}'
		np.savez(f'{fname_out}.npz', **epoch_times)
		np.savez(f'{fname_out}_{EPOCH_LENGTH}s_random.npz', **random_epochs)


if __name__ == "__main__":
	main()