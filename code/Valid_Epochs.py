import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding'

def get_epochs(time, speed, threshold):
	run_bool=speed>threshold
	start_times=np.array([])
	stop_times=np.array([])
	delta_indices=np.argwhere(np.diff(run_bool))
	for i in delta_indices:
	    if speed[i+1]>threshold:
	        start_times=np.append(start_times,time[i+1])
	    else:
	        stop_times=np.append(stop_times,time[i+1])
	return start_times, stop_times

#Make sure epoch lengths are in order least to greatest
def get_behavioral_epochs(epoch_lengths):
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	all_epoch_data={}
	for e in epoch_lengths:
	    all_epoch_data[f'{e}s_epochs']=[{},{}]

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
	    series_data=np.load(f'{PROJECT_PATH}\\data\\np files\\behavioral series\\{session_id}')
	    spon_time,y=series_data['time'],series_data['filtered_speed']

	    start_times, stop_times=get_epochs(spon_time,y,5)
	    #print(len(start_times),len(stop_times))
	    if len(start_times)==len(stop_times):
	        length=min(len(start_times),len(stop_times))-1
	    else:
	        length=min(len(start_times),len(stop_times))
	    
	    #Identify all valid running/stationary epochs for each entered epoch length
	    for e in epoch_lengths:
	        valid_run_epochs=[]
	        valid_sta_epochs=[]
	        if start_times[0]>stop_times[0]:
	            for i in range(length):
	                if (stop_times[i+1]-start_times[i])>e:
	                    valid_run_epochs.append([start_times[i],stop_times[i+1]])
	                elif (start_times[i]-stop_times[i])>e:
	                    valid_sta_epochs.append([stop_times[i],start_times[i]])
	        else:
	            for i in range(length):
	                if (stop_times[i]-start_times[i])>e:
	                    valid_run_epochs.append([start_times[i],stop_times[i]])
	                elif (start_times[i+1]-stop_times[i])>e:
	                    valid_sta_epochs.append([stop_times[i],start_times[i+1]])

	        all_epoch_data[f'{e}s_epochs'][0][session_id]=valid_sta_epochs
	        all_epoch_data[f'{e}s_epochs'][1][session_id]=valid_run_epochs

	    #print(all_epoch_data)

	return all_epoch_data