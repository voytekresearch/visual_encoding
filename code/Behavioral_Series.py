import os
import numpy as np
import pandas as pd
from scipy import signal
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH = 'C:/users/micha/visual_encoding' # 'C:\\Users\\User\\visual_encoding'
FS = 2500

# import cusgtom functions
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_running_timeseries

#Make sure epoch lengths are in order least to greatest
def main():
	# identify / create directories
	dir_output = PROJECT_PATH + '/data/behavior/running'
	if not os.path.exists(dir_output): 
		os.mkdirs(dir_output)

	# load project cache
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	for session_id in [732592105]:#sessions[sessions.get('session_type')=='functional_connectivity'].index:
		# get session data and display progress	
		print(f'analyzing session: \t{session_id}')
		session = cache.get_session_data(session_id)

		# Create uniform set of data using interpolation and save to file
		time, velocity = get_running_timeseries(session, FS)
		np.savez(f'{dir_output}/running_{session_id}', time=time, velocity=velocity)

		# save running data for spontaneous epoch to file
		fname_out = dir_output + f'/running_{session_id}_spont'
		save_spontaneous_epoch(session, time, velocity, fname_out)

def save_spontaneous_epoch(session, new_t, new_x, fname_out):    
	# Isolating the largest timeframe of spontaneous activity
	stimuli_df=session.stimulus_presentations
	stimuli_df=stimuli_df[stimuli_df.get('stimulus_name')=='spontaneous'].get(['start_time','stop_time'])
	# stimuli_df=stimuli_df[stimuli_df.get('start_time')>3600]
	stimuli_df=stimuli_df.assign(diff=stimuli_df.get('stop_time')-stimuli_df.get('start_time')).sort_values(by='diff',ascending=False)
	start_time=stimuli_df.get('start_time').iloc[0]
	stop_time=stimuli_df.get('stop_time').iloc[0]

	# epoch data
	epoch_mask = (new_t>start_time) & (new_t<stop_time)
	spon_time = new_t[epoch_mask]
	spon_speed = new_x[epoch_mask]

	# Applying a median filter
	N = 501
	# y = signal.medfilt(spon_speed, [N])
	y=spon_speed
	# Save filtered data
	np.savez(fname_out, time=spon_time,filtered_speed=y)

if __name__ == "__main__":
	main()