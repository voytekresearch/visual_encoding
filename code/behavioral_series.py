import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
FS = 2500 # sampling frequency for interpolation
SMOOTH = True # whether to smooth data (median filter)
KERNEL_SIZE = 1*FS # kernel size for median filter
STIMULUS_NAME = 'spontaneous' # name of stimulus in allen dataset
TRIAL = 4 # trial of stimulus

# import custom functions
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_running_timeseries
from allen_vc.epoch_extraction_tools import get_stimulus_behavioral_series

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

	# Iterate over each session
	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
		# get session data and display progress	
		print(f'Analyzing session: \t{session_id}')
		session = cache.get_session_data(session_id)

		# create uniform set of data using interpolation and save to file
		time, velocity = get_running_timeseries(session, FS)
		np.savez(f'{dir_output}/running_{session_id}', time=time, velocity=velocity)

		# save running data for spontaneous epoch to file
		fname_out = dir_output + f'/running_{session_id}_{STIMULUS_NAME}_{TRIAL}'
		results = get_stimulus_behavioral_series(STIMULUS_NAME, session, time, velocity, \
			trial=TRIAL, smooth=SMOOTH, kernel_size=KERNEL_SIZE)
		np.savez(fname_out, time=results[0], velocity_raw=results[1], velocity=results[2])


if __name__ == "__main__":
	main()