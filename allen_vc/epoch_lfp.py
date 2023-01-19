"""
Epoch LFP arond stimulus presentation time

"""

# imports
import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from time import time as timer
from time import ctime as time_now
from utils import find_probes_in_region, hour_min_sec, save_pkl

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_OUT = "data/lfp_data/lfp_epochs/natural_movie" # where to save output relative to both paths above

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest
REGION = "VISp" # brain structure of interest

# settings - stimulus epoch of interest
STIM_PARAMS = dict({
    'stimulus_name' : 'natural_movie_one_more_repeats',
    'frame' : 0
    }) # other stim params
T_WINDOW = [0, 30]  # epoch bounds (sec) [time_before_stim, tiimem_aftfer_stim]
STIM_CODE = 'movie' # alternate stimulus name for output filename

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(f'{dir_results}/npy'): os.makedirs(dir_results)
        if not os.path.exists(f'{dir_results}/pkl'): os.makedirs(dir_results)
    
    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")

    # get session info for dataset of interest
    sessions_all = cache.get_session_table()
    session_ids = sessions_all[sessions_all['session_type']==SESSION_TYPE].index.values

    # loo[ through all session for dataset of interest
    for i_session, session_id in enumerate(session_ids):
        # display progress
        t_start_s = timer()
        print(f"\n\n Beginning session {i_session+1}/{len(session_ids)}: \t{time_now()}")
        print(f"    session ID: {session_id}")

        # load session data
        session = cache.get_session_data(session_id)

        # get stim info for session
        stim_table = session.stimulus_presentations
        for param_name in STIM_PARAMS.keys():
            stim_table = stim_table[stim_table[param_name] == STIM_PARAMS[param_name]]

        # get probe info (for region of interest)
        if REGION is None:
            probe_ids = session.probes.index.values
        else:
            probe_ids, _ = find_probes_in_region(session, REGION)

        # display progress
        print(f"    {len(probe_ids)} probe(s) in ROI")
            
        # loop through all probes for region of interst
        for probe_id in probe_ids:
            # skip probes with no LFP data
            if ~ session.probes.loc[probe_id, 'has_lfp_data']:
                print(f"    No LFP data for probe: {probe_id}... skipping")
                continue
                
            # load LFP data
            print(f'    importing LFP data for probe: {probe_id}')
            lfp = session.get_lfp(probe_id)

            # get LFP for ROI
            if ~ (REGION is None):
                chan_ids = session.channels[(session.channels.probe_id==probe_id) & \
                    (session.channels.ecephys_structure_acronym==REGION)].index.values
                lfp = lfp.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))

            # epoch LFP data around stimulus
            print(f'    aligning to stimulus')
            lfp_a, time = align_lfp(lfp, stim_table.start_time.values, 
                stim_table.index.values, t_window=T_WINDOW, dt=1/FS)

            # create Neo object
            print('    creating Neo object')
            t_start = stim_table.start_time.values + T_WINDOW[0]
            block = create_neo_block(lfp_a, FS, t_start)

            # save results
            print('    saving data')
            fname_out = f"{session_id}_{probe_id}_lfp_{STIM_CODE}"
            for base_path in [PROJECT_PATH, MANIFEST_PATH]:
                dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
                
                # save lfp array as .npz
                np.savez(f"{dir_results}/npy/{fname_out}.npz", lfp=lfp_a, time=time) 

                # save Neo object as .pkl
                save_pkl(block, f"{dir_results}/pkl/{fname_out}.pkl")

        # display progress
        _, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    session complete in {min} min and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def align_lfp(lfp, t_stim, ids, t_window=[-1,1], dt=0.001):
    trial_window = np.arange(t_window[0], t_window[1], dt)
    time_selection = np.concatenate([trial_window + t for t in t_stim])

    inds = pd.MultiIndex.from_product((ids, trial_window), 
                                    names=('presentation_id', 'time_from_presentation_onset'))

    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name='aligned_lfp')
    ds = ds.assign(time=inds).unstack('time')

    aligned_lfp = ds['aligned_lfp']

    return aligned_lfp, trial_window

def create_neo_block(lfp, fs, t_start, block_name=None, units='uV'):
    """
    Creates a Neo Block object from an LFP array and its associated timestamps. 
    A Neo Segment is created within the Block for each trial.

    Parameters
    ----------
    lfp : array_like
        A 3D array of LFP data in the unconventional order (n_channels, n_trials, n_samples).
    fs : float
        Sampling rate of the LFP data.
    t_start : array_like
        Array of shape (n_trials,) of timestamps corresponding to the start of each 
        epoch in `lfp`.
    block_name : str, optional
        Name of the block. If None, no name is assigned.
    units : str, optional
        Units of the LFP data. Default is 'uV'.

    Returns
    -------
    block : neo.core.Block
        Neo Block object containing the LFP data.
    """

    # imports
    from neo.core import Block, Segment, AnalogSignal
    import quantities as pq
    
    # create Neo Block object
    if block_name is None:
        block = Block()
    else:
        block = Block(name=block_name)

    # create Neo Segment for each trial
    for epoch in range(lfp.shape[1]):
        segment = Segment(name=f'trial_{epoch}')
        block.segments.append(segment)

        # add LFP data
        lfp_as = AnalogSignal(lfp[:,epoch].T, units=units, sampling_rate=fs*pq.Hz, 
            t_start=t_start[epoch]*pq.s)
        segment.analogsignals.append(lfp_as)

    return block


if __name__ == '__main__':
    main()