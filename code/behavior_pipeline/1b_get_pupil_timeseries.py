"""
Compute velocity time-series for each session in a given dataset.
"""

# imports
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# settings - directories
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
MANIFEST_PATH = "C:/datasets/allen_vc/manifest_files" # local dataset directory

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest

# settings - dataset details
PF = 50 # Pupil area interpolation frequency

#Make sure epoch lengths are in order least to greatest
def main():
	# Define/create directories for output
	dir_results = f"{PROJECT_PATH}/data/behavior/pupil/session_timeseries"
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
		time, velocity = get_pupil_timeseries(session, PF)

		# save to file
		if velocity is not None:
			np.savez(f'{dir_results}/pupil_area_{session_id}', time=time, velocity=velocity)


def get_pupil_timeseries(session, pf):
	"""
	load screen gaze data for a given session. pupil area data are interpolated
	to a sampling frequnecy of 'pf.'

	Parameters
	----------
	session : AllenSDK session object
		AllenSDK session object.
	pf : int
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
	pupil_data = session.get_screen_gaze_data()

	if pupil_data is None:
		return None, None

	pupil_data = pupil_data[pupil_data['raw_pupil_area'].notna()]
	values = pupil_data['raw_pupil_area']
	time_points = pupil_data.index
	
	#Create uniform set of data using interpolation
	model = interpolate.interp1d(time_points, values)
	time = np.arange(time_points[0], time_points[-1], 1/pf)
	velocity = model(time)

	return time, velocity


if __name__ == "__main__":
	main()