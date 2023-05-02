"""
Utility functions for AllenSDK data analysis.

"""

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
    unit_info = unit_info.get(['ecephys_session_id','specimen_id',\
        'ecephys_probe_id','ecephys_channel_id'])#.drop_duplicates()

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


def gen_neo_spiketrains(session_id, manifest_path, metadata, brain_structure=None):
    """
    load spiking data for a session and reformat as Neo object.

    Parameters
    ----------
    session : AllenSDK session object
        AllenSDK session object.
    manifest_path : str
        path to AllenSDK manifest file.
    metadata : Pandas DataFrame
        contains information for unit annotations.
    brain_structure : str, optional
        include to filter results by brain structure. The default is None.

    Returns
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object
    """

    # imports
    import neo
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    #Get all session info
    session = cache.get_session_data(session_id)

    # Create Neo SpikeTrain object
    spiketrains = []

    # Retrive raw spikes and save as a group containing a single Neo SpikeTrain
    if brain_structure:
        for unit in session.units[session.units.get('ecephys_structure_acronym')==brain_structure].index:
            annotations = dict(metadata.loc[unit])
            annotations['unit_id'], annotations['region'] = unit, brain_structure
            session_spikes = session.spike_times[unit]
            spiketrains.append(neo.SpikeTrain(times=session_spikes, \
                units='sec', t_stop=session_spikes[-1], **annotations))
    else:
        for unit in session.units.index:
            annotations = dict(metadata.loc[unit])
            annotations['unit_id'] = unit
            session_spikes = session.spike_times[unit]
            spiketrains.append(neo.SpikeTrain(times=session_spikes, \
                units='sec', t_stop=session_spikes[-1], **annotations))

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
