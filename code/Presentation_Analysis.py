import os
import numpy as np
from scipy.ndimage import median_filter
from Valid_Epochs import get_diffs
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings (see Behavioral_Series.py and Valid_Epochs.py for details on parameters)
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
FS = 2500 
SMOOTH = True 
KERNEL_SIZE = 1*FS
SPEED_THRESHOLD = 5 
MAX_BREAK = 1 

# import custom functions
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_running_timeseries

#Make sure epoch lengths are in order least to greatest
def main():
	# identify / create directories
	dir_output = PROJECT_PATH + '/data/behavior/natural_movie_one'
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

		# Load already downloaded running data
		series_data = np.load(f'{PROJECT_PATH}/data/behavior/running/running_{session_id}.npz')
		time, velocity = series_data['time'], series_data['velocity']

		for trial in range(2): # two trials of natural movie one exist for each session
			# save running data for spontaneous epoch to file
			fname_out = dir_output + f'/movie_running_{session_id}_{trial}'
			results = get_movie_epoch(session, time, velocity, trial, smooth=SMOOTH, kernel_size=KERNEL_SIZE)
			np.savez(fname_out, time=results[0], velocity_raw=results[1], velocity=results[2])

			# calculate threshold crossings and individual movie times
			start_times, stop_times = get_diffs(results[0], results[2], SPEED_THRESHOLD, MAX_BREAK)
			diffs = np.array(sorted(start_times+stop_times))
			movie_times = get_movie_times(session, trial)

			# filter diffs for each start/stop movie time and create/save boolean array
			movie_run_bool = []
			for t in range(len(movie_times)-1):
				if len(diffs[(diffs>movie_times[t]) & (diffs<movie_times[t+1])])==0:
					if diffs[diffs<movie_times[t]][-1] in start_times:
						movie_run_bool.append(True)
					else:
						movie_run_bool.append(False)
				else:
					movie_run_bool.append(None)

			# save running/movie information
			fname_out = dir_output + f'/movie_running_bool_{session_id}_{trial}'
			np.save(fname_out, movie_run_bool)

def get_movie_epoch(session, time, velocity, trial, smooth=True, kernel_size=None):
	"""
	Filters and returns running time series for both natural movie one trials
	of a session. 
	"""
	stimuli_df = session.get_stimulus_epochs()
	stimuli_df = stimuli_df[stimuli_df.get('stimulus_name')=='natural_movie_one_more_repeats'].get(['start_time','stop_time','stimulus_block'])
	stimuli_df = stimuli_df.sort_values(by='stimulus_block')
    
	start_time = stimuli_df.get('start_time').iloc[trial]
	stop_time = stimuli_df.get('stop_time').iloc[trial]
	epoch_mask = (time>start_time) & (time<stop_time)
	movie_time = time[epoch_mask]
	movie_speed = velocity[epoch_mask]

	# Apply a median filter
	if smooth:
			# make sure kernel size is odd
		if kernel_size is None:
			print("Please provide kernel_size")
		else:
			if kernel_size % 2 == 0:
				ks = kernel_size + 1
		# filter
		movie_speed_filt = median_filter(movie_speed, ks)
	else:
		movie_speed_filt = None

	return movie_time, movie_speed, movie_speed_filt

def get_movie_times(session, trial):
	"""
	For a given session, returns the start times of each individual movie.
	"""
	stimuli_df = session.stimulus_presentations
	stimuli_df = stimuli_df[stimuli_df.get('stimulus_name')=='natural_movie_one_more_repeats']
	starts = np.array(stimuli_df[stimuli_df.get('frame')==0].get('start_time'))[trial*30:(trial+1)*30]
	end = np.array(stimuli_df[stimuli_df.get('frame')==899].get('stop_time'))[((trial+1)*30)-1]
	return np.append(starts, end)


if __name__ == "__main__":
	main()