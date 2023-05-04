"""
Utility functions for AllenSDK data analysis.

"""

# imports
import numpy as np


def compute_running_speed(session, fs):
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
    running_speed : float, array
        running speed (interpolated).

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
    running_speed = model(time)

    return time, running_speed


def compute_pupil_area(session, fs):
	"""
	load screen gaze data for a given session. pupil area data are interpolated
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
	pupil_area : float, array
		pupil area (interpolated).

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
	time = np.arange(time_points[0], time_points[-1], 1/fs)
	pupil_area = model(time)

	return time, pupil_area


def create_neo_spiketrains(session, brain_structure=None):
    """
    load spiking data for a session and reformat as Neo object.

    Parameters
    ----------
    session : AllenSDK session object
        AllenSDK session object.
    brain_structure : str, optional
        include to filter results by brain structure. The default is None.

    Returns
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object
    """

    # imports
    from neo import SpikeTrain

    # Init list of Neo SpikeTrain objects
    spiketrains = []

    # Get unit info
    if brain_structure:
        unit_info = session.units[session.units.get('ecephys_structure_acronym')==brain_structure]
    else:
        unit_info = session.units

    # Loop through units and create Neo SpikeTrain objects
    for _, row in unit_info.iterrows():
        # Get spike times
        spike_times = session.spike_times[row.name]

        # Create Neo SpikeTrain object
        spiketrain = SpikeTrain(times=spike_times, units='sec', t_stop=spike_times[-1], \
            name=row.name)
        spiketrain.annotate(unit_id=row.name, probe_id=row['probe_id'], \
            channel_id=row['peak_channel_id'], brain_structure=row['ecephys_structure_acronym'])
        spiketrains.append(spiketrain)

    return spiketrains


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


def find_probes_in_region(session, region):
    probe_ids = session.probes.index.values
    has_region = np.zeros_like(probe_ids).astype(bool)

    for i_probe, probe_id in enumerate(probe_ids):
        regions = session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique()
        has_region[i_probe] = region in regions

    ids = probe_ids[has_region]
    names = session.probes.description.values[has_region]

    return ids, names


def align_lfp(lfp, t_stim, t_window=[-1,1], dt=0.001):
    """
    Modified from AllenSDK example code:
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_lfp_analysis.html

    Aligns LFP data to stimulus presentation times.

    Parameters
    ----------
    lfp : xarray.core.dataarray.DataArray
        LFP data to be aligned. Must have a time coordinate.
    t_stim : array_like
        Array of shape (n_trials,) of timestamps corresponding to the start of each
        epoch in `lfp`.
    t_window : array_like, optional
        Array of shape (2,) of the time window (in seconds) to be extracted around
        each stimulus presentation. Default is [-1,1].
    dt : float, optional
        Time resolution (in seconds) of the aligned LFP data. Default is 0.001.

    Returns
    -------
    aligned_lfp : xarray.core.dataarray.DataArray
        LFP data aligned to stimulus presentation times. (n_trials, n_channels, n_timepoints)
    trial_window : array_like
        Array of shape (n_timepoints,) of the time window (in seconds) around each
        stimulus presentation.
    """

    # imports
    import pandas as pd

    # determine indices of time window around stimulus presentation
    trial_window = np.arange(t_window[0], t_window[1], dt)
    time_selection = np.concatenate([trial_window + t for t in t_stim])
    inds = pd.MultiIndex.from_product((np.arange(len(t_stim)), trial_window), 
                                    names=('presentation_id', 'time_from_presentation_onset'))

    # epoch LFP data around stimulus presentation
    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name='aligned_lfp')
    ds = ds.assign(time=inds).unstack('time')

    # reshape data (n_trials, n_channels, n_timepoints)
    aligned_lfp = ds['aligned_lfp']
    aligned_lfp = np.swapaxes(aligned_lfp, 0, 1)

    return aligned_lfp, trial_window

