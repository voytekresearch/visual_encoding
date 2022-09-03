# imports
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def get_unit_info(manifest_path, brain_structure=None, session_type=None):

	# Create Allensdk cache object
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	# Get all unit info
	unit_info = cache.get_units()

	# filter by session type
	if session_type:
		unit_info = unit_info[unit_info.get('session_type')==session_type]

	# filter by brain structure
	if brain_structure:
		unit_info = unit_info[unit_info.get('ecephys_structure_acronym')==brain_structure]

	# get info of interest
	unit_info = unit_info.get(['ecephys_session_id','specimen_id','ecephys_probe_id','ecephys_channel_id']).drop_duplicates()

	return unit_info

def get_running_timeseries(session, fs):
	# imports
	from scipy import interpolate

	# get running data
	run_data=session.running_speed
	run_data=run_data.assign(mid_time=(run_data.get('start_time')+run_data.get('end_time'))/2)
	time_points = np.array(run_data.get('mid_time'))
	values = np.array(run_data.get('velocity'))
	
	#Create uniform set of data using interpolation
	model = interpolate.interp1d(time_points, values)
	time = np.arange(time_points[0], time_points[-1], 1/fs)
	signal = model(time)

	return time, signal