"""
Compute velocity time-series for each session in a given dataset.
"""

# imports
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# settings - directories
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
MANIFEST_PATH = "D:/datasets/allen_vc" # local dataset directory

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest

# settings - dataset details
FS = 1250 # LFP sampling freq

#Make sure epoch lengths are in order least to greatest
def main():
	# Define/create directories for output
	dir_results = f"{PROJECT_PATH}/data/behavior/running/session_timeseries"
	if not os.path.exists(dir_results): 
		os.makedirs(dir_results)

	# load project cache
	cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")
	sessions = cache.get_session_table()

	# Iterate over each session
	for session_id in sessions[sessions.get('session_type')==SESSION_TYPE].index:
		# get session data and display progress	
		print(f'Analyzing session: \t{session_id}')
		session = cache.get_session_data(session_id)

		# create uniform set of data using interpolation
		time, velocity = get_running_timeseries(session, FS)

		# save to file
		np.savez(f'{dir_results}/running_{session_id}', time=time, velocity=velocity)


def get_running_timeseries(session, fs):
	"""
	load running wheel data for a given session. velocity data are interpolated
	to a sampling frequnecy of 'fs.'

	Parameters
	----------
	session : AllenSDK session object
		AllenSDK session object.
	fs : int
		sampling frequency.

	Returns
	-------
	time : float, array
		time vector.
	velocity : float, array
		velocity (interpolated).

	"""
    
	# imports
	from scipy import interpolate

	# get running data
	run_data = session.running_speed
	run_data = run_data.assign(mid_time=(run_data.get('start_time')+run_data.get('end_time'))/2)
	time_points = np.array(run_data.get('mid_time'))
	values = np.array(run_data.get('velocity'))
	
	#Create uniform set of data using interpolation
	model = interpolate.interp1d(time_points, values)
	time = np.arange(time_points[0], time_points[-1], 1/fs)
	velocity = model(time)

	return time, velocity


if __name__ == "__main__":
	main()