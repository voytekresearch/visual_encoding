import os
import numpy as np
import pandas as pd
import random
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC = f'{PROJECT_PATH}/data/behavior/running'
SPEED_THRESHOLD = 5 # Any speed greater than this value is considered running (cm/s)
MIN_DURATION = 1 # Minimum time of measured epochs (s)
EPOCH_LENGTH = 10
STIMULUS_NAME = 'spontaneous'
TRIAL = 4 
FS = 2500 # Sampling rate (Hz)

# Import custom functions			  
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_valid_epochs
from allen_vc.epoch_extraction_tools import get_epoch_times, get_random_epoch

def main():

	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	# Initialize storage space
	epoch_times = {}
	random_epochs = {}

	# Iterate over all sessions
	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
		print(f'Analyzing Session:      {session_id}')
		series_data = np.load(f'{DATA_LOC}\\running_{session_id}_{STIMULUS_NAME}_{TRIAL}.npz')

		# Calculate/store above and below epochs
		running_epochs, stationary_epochs = \
		get_epoch_times(series_data['velocity'], SPEED_THRESHOLD, MIN_DURATION*FS)
		
		epoch_times[f'{session_id}_running'] = series_data['time'][0] + running_epochs/FS
		epoch_times[f'{session_id}_stationary'] = series_data['time'][0] + stationary_epochs/FS

		# Choose and store random epoch from selection cropped to EPOCH_LENGTH
		random.seed(101)
		random_epochs[f'{session_id}_running'] = get_random_epoch(\
			epoch_times[f'{session_id}_running'], EPOCH_LENGTH)
		random_epochs[f'{session_id}_stationary'] = get_random_epoch(\
			epoch_times[f'{session_id}_stationary'], EPOCH_LENGTH)

	# Save data
	np.savez(f'{DATA_LOC}/{STIMULUS_NAME}_running_epochs.npz', **epoch_times)
	np.savez(f'{DATA_LOC}/{STIMULUS_NAME}_random_running_epoch_{EPOCH_LENGTH}s.npz', **random_epochs)


if __name__ == "__main__":
	main()