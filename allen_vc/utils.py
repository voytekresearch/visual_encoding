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


def gen_pop_spiketrain(spike_trains, units='s', t_stop=None):
    """Generates a population spiketrain from a list of individual spike trains.

    Parameters
    ----------
    spike_trains : list
        A list of Neo SpikeTrains 
    units : str, optional
        The units of the spike times (default is 's')
    t_stop : float, optional
        The stop time of the population spike train. If not provided, it will be
        set to the last spike time (default is None)

    Returns
    -------
    pop_spiketrain : neo.SpikeTrain
        A Neo SpikeTrain object with the population spike train
    """
    
    # imports
    import neo

    # concatenate spike trains across population
    pop_spikes = np.sort(np.array(np.concatenate(np.array(spike_trains, dtype=object))))

    # get stop time
    if t_stop is None:
        t_stop = pop_spikes[-1]

    # create Neo SpikeTrain for population
    pop_spiketrain = neo.SpikeTrain(pop_spikes, units=units, t_stop=t_stop)
    
    return pop_spiketrain


def hour_min_sec(duration):
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs


def find_probes_in_region(session, region):
    probe_ids = session.probes.index.values
    has_region = np.zeros_like(probe_ids).astype(bool)

    for i_probe, probe_id in enumerate(probe_ids):
        regions = session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique()
        has_region[i_probe] = region in regions

    ids = probe_ids[has_region]
    names = session.probes.description.values[has_region]

    return ids, names

def save_pkl(dictionary, fname_out, method='pickle'):
    """
    save dictionary as pickle

    Parameters
    ----------
    dictionary : dict
        dictionary to be saved to file.
    fname_out : str
        filename for output.
    method : str, optional
        what method (pickle or joblib) to use to saving. 
        OS users were having issues saving large files with pickle.
        The default is 'pickle'.

    Returns
    -------
    None.

    """
    # imports
    import pickle

    # save pickle
    f = open(fname_out, "wb")
    pickle.dump(dictionary, f)
    f.close()

def get_neo_group_names(block):
    """
    Get the names of all groups in a Neo block.
    
    Parameters
    ----------
    block : neo.Block
        Neo block containing groups.

    Returns
    -------
    group_names : list
        List of group names.   
    """

    group_names = []
    for group in block.groups:
        group_names.append(group.name)
    
    return group_names


def params_to_df(params, max_peaks):
    # get per params
    df_per = pd.DataFrame(params.get_params('peak'),
        columns=['cf','pw','bw','idx'])

    # get ap parmas
    df_ap = pd.DataFrame(params.get_params('aperiodic'),  
        columns=['offset', 'knee', 'exponent'])

    # get quality metrics
    df_ap['r_squared'] = params.get_params('r_squared')

    # initiate combined df
    df = df_ap.copy()
    columns = []
    for ii in range(max_peaks):
        columns.append([f'cf_{ii}',f'pw_{ii}',f'bw_{ii}'])
    df_init = pd.DataFrame(columns=np.ravel(columns))
    df = df.join(df_init)

    # app peak params for each peak fouond
    for i_row in range(len(df)):
        # check if row had peaks
        if df.index[ii] in df_per['idx']:
            # get peak info for row
            df_ii = df_per.loc[df_per['idx']==i_row].reset_index()
            # loop through peaks
            for i_peak in range(len(df_ii)):
                # add peak info to df
                for var_str in ['cf','pw','bw']:
                    df.at[i_row, f'{var_str}_{i_peak}'] = df_ii.at[i_peak, var_str]
    
    return df
    

def combine_spike_lfp_dfs(spike_df, lfp_df, ses_id, region, state=None):
    '''
    Combines spike and LFP dataframes given session ID and region.

    Parameters
    ----------
    spike_df : dataframe
        Dataframe containing spike parameters.
    lfp_df : dataframe
        Dataframe containing LFP parameters.
    ses_id : int
        Session ID for dataframes.
    region : str
        Brain region for dataframes.
    state : str, optional
        Running state for dataframes.

    Returns
    -------
    df : dataframe
        Combined dataframe of spike and LFP parameters.
    '''

    # filter region
    reg_spike_df = spike_df[spike_df.get("brain_structure")==region]

    # filter session
    ses_spike_df = reg_spike_df[reg_spike_df.get("session")==ses_id]
    ses_lfp_df = lfp_df[lfp_df.get("session")==ses_id]
    
    # take median across channels for param data
    chan_med_lfp_df = channel_medians(ses_lfp_df, ses_lfp_df.columns).drop(columns="chan_idx")
    
    if state is not None:
        state_spike_df = ses_spike_df[ses_spike_df.get("running")==state]
        return chan_med_lfp_df.merge(state_spike_df)
    
    return chan_med_lfp_df.merge(ses_spike_df)


def channel_medians(lfp_df, col_names):
    """
    This function takes in a dataframe of LFP data and a list of column names and returns a dataframe of the medians for each channel for each epoch.

    Parameters
    ----------
    lfp_df : dataframe
        A dataframe of LFP data
    col_names : list
        A list of column names

    Returns
    -------
    dataframe
        A dataframe of the medians for each channel for each epoch
    """
    
    #imports
    import pandas as pd
    
    medians = [0]*len(col_names)
    for epoch in lfp_df.get("epoch_idx").unique():
        epoch_df = lfp_df[lfp_df.get("epoch_idx")==epoch]
        medians = np.vstack((medians, epoch_df.median()))

    medians = np.delete(medians, (0), axis=0)
    return pd.DataFrame(data = medians, columns = col_names)#.drop(columns = 'chan_idx')