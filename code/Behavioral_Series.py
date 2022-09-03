import os
import numpy as np
from scipy import signal
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH = 'C:/users/micha/visual_encoding' # 'C:\\Users\\User\\visual_encoding'
FS = 2500
KERNEL_SIZE = 501

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

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
		# get session data and display progress	
		print(f'Analyzing session: \t{session_id}')
		session = cache.get_session_data(session_id)

		# Create uniform set of data using interpolation and save to file
		time, velocity = get_running_timeseries(session, FS)
		np.savez(f'{dir_output}/running_{session_id}', time=time, velocity=velocity)

		# save running data for spontaneous epoch to file
		fname_out = dir_output + f'/running_{session_id}_spont'
		save_spontaneous_epoch(session, time, velocity, fname_out)

def save_spontaneous_epoch(session, time, velocity, fname_out):    
	# Isolate the largest timeframe of spontaneous activity
	stimuli_df = session.stimulus_presentations
	stimuli_df = stimuli_df[stimuli_df.get('stimulus_name')=='spontaneous'].get(['start_time','stop_time'])
	# stimuli_df=stimuli_df[stimuli_df.get('start_time')>3600]
	stimuli_df = stimuli_df.assign(diff=stimuli_df.get('stop_time')-stimuli_df.get('start_time')).sort_values(by='diff',ascending=False)
	start_time = stimuli_df.get('start_time').iloc[0]
	stop_time = stimuli_df.get('stop_time').iloc[0]

	# epoch data
	epoch_mask = (time>start_time) & (time<stop_time)
	spont_time = time[epoch_mask]
	spont_speed = velocity[epoch_mask]

	# Apply a median filter
	spont_speed_filt = signal.medfilt(spont_speed, [KERNEL_SIZE])

	# Save filtered data
	np.savez(fname_out, time=spont_time, velocity=spont_speed_filt)

if __name__ == "__main__":
	main()