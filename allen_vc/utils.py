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

def get_spiking_data(session, manifest_path, brain_structure=None):
	"""
	load and save spiking data of given units. include metrics spike_times,
	spike_amplitudes, and mean_waveforms.

	Parameters
	----------
	session : AllenSDK session object
		AllenSDK session object.
	manifest_path : str
		path to AllenSDK manifest file.
	brain_structure : str, optional
		include to filter results by brain structure. The default is None.

	Returns
	-------
	spike_times : dict
		keys as unit ids and respective spike times as values.
	spike_amplitudes : dict
		keys as unit ids and respective spike amplitudes as values.
	mean_waveforms : dict
		keys as unit ids and respective mean_waveforms as values.

	"""

	# imports
	from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

	# Create Allensdk cache object
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	# Initialize storage dictionaries
	spike_times = {}
    spike_amplitudes = {}
    mean_waveforms = {}

    #Get all session info
    session = cache.get_session_data(ses)

    #Filter for brain structure if argument included
    if brain_structure:
    	for unit in session.units[session.units.get('ecephys_structure_acronym')==brain_structure].index:
            spike_times[unit] = session.spike_times[unit]
            spike_amplitudes[unit] = session.spike_amplitudes[unit]
            mean_waveforms[unit] = session.mean_waveforms[unit]
    else:
    	for unit in session.units.index:
    		spike_times[unit] = session.spike_times[unit]
            spike_amplitudes[unit] = session.spike_amplitudes[unit]
            mean_waveforms[unit] = session.mean_waveforms[unit]

	return spike_times, spike_amplitudes, mean_waveforms

def get_valid_epochs(start_times, stop_times, epoch_length):
	"""
	filter epochs as behavior positive/negative based on a series of start
	and stop times. return epochs greater than a given epoch length.

	Parameters
	----------
	start_times : list/array
		epoch start times.
	stop_times : list/array
		epoch stop times.
	epoch_length : int
		length of epoch to be found.

	Returns
	-------
	positive_epochs : list
		epochs > epoch length that are behavior positive
	negative_epochs : list
		epochs > epoch length that are behavior negative

	"""
	#Determine the amount of times to iterate through series to include all data
	if len(start_times)==len(stop_times):
	        iter_length = min(len(start_times),len(stop_times))-1
	    else:
	        iter_length = min(len(start_times),len(stop_times))

	#Identify all valid running/stationary epochs for each entered epoch length
	positive_epochs = []
	negative_epochs = []
	
	if start_times[0]>stop_times[0]:
	    for i in range(iter_length):
	        if (stop_times[i+1]-start_times[i])>e:
	            positive_epochs.append([start_times[i],stop_times[i+1]])
	        elif (start_times[i]-stop_times[i])>e:
	            negative_epochs.append([stop_times[i],start_times[i]])
	else:
	    for i in range(iter_length):
	        if (stop_times[i]-start_times[i])>e:
	           positive_epochs.append([start_times[i],stop_times[i]])
	        elif (start_times[i+1]-stop_times[i])>e:
	            negative_epochs.append([stop_times[i],start_times[i+1]])

	return positive_epochs, negative_epochs
		