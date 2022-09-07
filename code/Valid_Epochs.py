import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC=f'{PROJECT_PATH}\\data\\epoch_data'
epoch_lengths = [10,20,30]
speed_threshold = 5

import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_valid_epochs

def main():

	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
		print(f'Analyzing Session:      {session_id}')
		series_data = np.load(f'{PROJECT_PATH}\\data\\behavior\\running\\running_{session_id}_spont.npz')
		spon_time, speed = series_data['time'],series_data['velocity']

		start_times, stop_times = get_epochs(spon_time,speed,speed_threshold)
	    
		#Identify all valid running/stationary epochs for each entered epoch length
		for epoch_length in epoch_lengths:
			valid_run_epochs, valid_sta_epochs = get_valid_epochs(start_times, stop_times, epoch_length)

			#Choose random epoch to examine
			#np.random.seed(101)

			rand_epochs = []
			for valid_epochs in [valid_sta_epochs, valid_run_epochs]:
				if valid_epochs:
					rand = valid_epochs[np.random.choice(len(valid_epochs))]
					rand_epochs.append(np.array([rand[0],rand[0]+epoch_length]))
				else:
					rand_epochs.append(np.array([]))

			#Save all valid epochs and randomly chosen epoch
			np.savez(f'{DATA_LOC}\\{session_id}_all_{epoch_length}s_valid_behavioral_epochs.npz', stationary=valid_sta_epochs, running=valid_run_epochs)
			np.savez(f'{DATA_LOC}\\{session_id}_{epoch_length}s_random_epochs.npz', stationary=rand_epochs[0], running=rand_epochs[1])

def get_epochs(time, speed, threshold):
	#Identify time points where subject changes behavior
	#Start times indicate when behavioral value > threshold
	run_bool = speed>threshold
	start_times = []
	stop_times = []
	delta_indices = np.argwhere(np.diff(run_bool))
	for i in np.stack(delta_indices, axis=-1)[0]:
		if speed[i+1]>threshold:
			start_times.append(time[i+1])
		else:
			stop_times.append(time[i+1])
	return start_times, stop_times

if __name__ == "__main__":
	main()