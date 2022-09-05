import os
import numpy as np
import pandas as pd
from allen_vc.utils import get_valid_epochs
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC=f'{PROJECT_PATH}\\data\\epoch_data'
epoch_lengths = [10,20,30]
speed_threshold = 5

def main():

	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
	    series_data = np.load(f'{PROJECT_PATH}\\data\\behavior\\running\\running_{session_id}_spont')
	    spon_time, speed = series_data['time'],series_data['velocity']

	    start_times, stop_times = get_epochs(spon_time,speed,speed_threshold)
	    
	    #Identify all valid running/stationary epochs for each entered epoch length
	    for e in epoch_lengths:
	        valid_run_epochs, valid_sta_epochs = get_valid_epochs(start_times, stop_times, e)

	        #Choose random epoch to examine
	        np.random.seed(101)

	        sta_rand = np.random.choice(valid_sta_epochs)
	        run_rand = np.random.choice(valid_run_epochs)

	        sta_epoch = np.array([sta_rand[0],sta_rand[0]+e])
	        run_epoch = np.array([run_rand[0],run_rand[0]+e])

	        #Save all valid epochs and randomly chosen epoch
		    np.savez(f'{DATA_LOC}\\{session_id}_all_{e}s_valid_behavioral_epochs.npz', stationary=valid_sta_epochs, running=valid_run_epochs)
		    np.savez(f'{DATA_LOC}\\{session_id}_{e}s_random_epoch.npz', stationary=sta_epoch, running=run_epoch)

def get_epochs(time, speed, threshold):
	#Identify time points where subject changes behavior
	#Start times indicate when behavioral value > threshold
	run_bool = speed>threshold
	start_times = []
	stop_times = []
	delta_indices = np.argwhere(np.diff(run_bool))
	for i in delta_indices:
	    if speed[i+1]>threshold:
	        start_times.append(time[i+1])
	    else:
	        stop_times.append(time[i+1])
	return start_times, stop_times

if __name__ == "__main__":
	main()