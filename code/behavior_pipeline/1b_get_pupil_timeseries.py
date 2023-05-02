"""
Compute velocity time-series for each session in a given dataset.
"""

# imports
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# imports- custom
import sys
sys.path.append('allen_vc')
from allen_utils import get_pupil_timeseries

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


if __name__ == "__main__":
	main()