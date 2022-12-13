# imports
import numpy as np
import pandas as pd

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

def get_spiking_data(session_id, manifest_path, brain_structure=None):
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
	session = cache.get_session_data(session_id)

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

def calculate_spike_metrics(raw_spikes, epoch):
	"""
	calculate spike metrics (mean firing rate, coefficient of variance, 
	SPIKE-distance, SPIKE-synchrony, and correlation coefficient) within
	a specified epoch given a matrix of spike times.

	Parameters
	----------
	raw_spikes: list/array
		list/array of lists/arrays of raw spike times for several units.
	epoch: list/array
		list of length 2 that includes epoch start and stop times.

	Returns
	-------
	mean_firing_rate: float
		mean firing rate over all units during specified epoch.
	coeff_of_var: float
		coefficient of variation over all units during specified epoch.
	spike_dist: float
		SPIKE-distance (pyspike) over all units during specified epoch.
	spike_sync: float
		SPIKE-synchrony (pyspike) over all units during specified epoch.
	corr_coeff:
		correlation coefficient (elephant) over all units during 
		specified epoch. 
	"""
	#Imports
	import pyspike as spk
	import neo
	import elephant
	import quantities as pq

	#Compute coefficient of variation
	def comp_cov(pop_spikes):
		isi = np.diff(pop_spikes)
		cov = np.std(isi) / np.mean(isi)   
		return cov

	#Store pyspike.SpikeTrain and Neo.SpikeTrain objects
	spk_trains = []
	neo_trains = []

	#Initialize metric calculation
	epoch_length = int(epoch[1]-epoch[0])
	pop_spikes = epoch[0]
	raw_spikes = [raw_spike[(raw_spike>epoch[0]) & (raw_spike<epoch[1])] for raw_spike in raw_spikes]
	fr=0

	for i in range(len(raw_spikes)):
		fr+=len(raw_spikes[i])/epoch_length
		pop_spikes = np.hstack([pop_spikes,raw_spikes[i]])
		spk_trains.append(spk.SpikeTrain(raw_spikes[i], epoch))
		neo_obj=neo.SpikeTrain(times=raw_spikes[i], units='sec', t_start=epoch[0], t_stop=epoch[1])
		neo_trains.append(neo_obj)

	mean_firing_rate = (fr/len(raw_spikes))
	pop_spikes = np.sort(pop_spikes)
	coeff_of_var = (comp_cov(pop_spikes))
	spike_dist = (spk.spike_distance(spk_trains))
	spike_sync = (spk.spike_sync(spk_trains))
	corr_coeff = (elephant.spike_train_correlation.correlation_coefficient(elephant.conversion.BinnedSpikeTrain(neo_trains, bin_size=1 * pq.s)))

	return mean_firing_rate, coeff_of_var, spike_dist, spike_sync, corr_coeff

		