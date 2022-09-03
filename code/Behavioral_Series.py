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
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
		print(f'analyzing session: \t{session_id}')

	    #Create uniform set of data using interpolation
		session=cache.get_session_data(session_id)
		new_t, new_x = get_running_timeseries(session, FS)
	    
		#Isolating the largest timeframe of spontaneous activity
		stimuli_df=session.stimulus_presentations
		stimuli_df=stimuli_df[stimuli_df.get('stimulus_name')=='spontaneous'].get(['start_time','stop_time'])
		#stimuli_df=stimuli_df[stimuli_df.get('start_time')>3600]
		stimuli_df=stimuli_df.assign(diff=stimuli_df.get('stop_time')-stimuli_df.get('start_time')).sort_values(by='diff',ascending=False)
		start_time=stimuli_df.get('start_time').iloc[0]
		stop_time=stimuli_df.get('stop_time').iloc[0]
		spon_epoch=np.array([start_time,stop_time])
		#print(f'Start Time: {start_time}, Stop Time: {stop_time}')

		spon_time=new_t[(new_t>start_time)][new_t[(new_t>start_time)]<stop_time]
		spon_speed=new_x[(new_t>start_time)][new_t[(new_t>start_time)]<stop_time]

		# Applying a median filter
		N = 501
		y = signal.medfilt(spon_speed, [N])

		#Save filtered data
		np.savez(f'{PROJECT_PATH}\\data\\np files\\behavioral series\\{session_id}', time=spon_time,filtered_speed=y,spon_epoch=spon_epoch)

if __name__ == "__main__":
	main()