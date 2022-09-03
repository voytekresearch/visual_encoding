# imports
import numpy as np

def get_unit_info(manifest_path, brain_structure=None, session_type=None):
	"""
	get info about single-units, including session, subject, probe, and channel

	Parameters
	----------
	manifest_path : str
		path to AllenSDK manifest file.
	brain_structure : str, optional
		include to filter results by brain structure. The default is None.
	session_type : TYPE, optional
		include to filter data by session type. Options include:
			'brain_observatory_1.1' and 'functional_connectivity'
			The default is None.

	Returns
	-------
	unit_info : DataFrame, optional
		unit info DataFrame

	"""

	# imports
	from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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