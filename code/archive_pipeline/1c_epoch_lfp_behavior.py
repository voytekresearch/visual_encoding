"""
Epoch LFP during longest spontaneous epoch according to running/stationary behavior

"""
# Names/labels
STIM_CODE = 'spontaneous' # alternate stimulus name for output filename
BEHAVIOR_NAME = 'running'

# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
MANIFEST_PATH = "E:/datasets/allen_vc/manifest_files"
RELATIVE_PATH_OUT = f"data/lfp_data/lfp_epochs/{STIM_CODE}/{BEHAVIOR_NAME}" # where to save output relative to both paths above 

# imports
import os
import numpy as np
import pandas as pd
import quantities as pq
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from time import time as timer
from time import ctime as time_now

# imports - custom
import sys
sys.path.append("allen_vc")
from utils import find_probes_in_region, hour_min_sec, save_pkl
from epoch_extraction_tools import get_epoch_times, split_epochs

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest
REGION = "VISp" # brain structure of interest
BEHAVIOR_PATH = f"{PROJECT_PATH}/data/behavior/{BEHAVIOR_NAME}/{STIM_CODE}"
BLOCK_POS = 4

# settings - stimulus epoch of interest
THRESHOLD = 1 # Threshold for identifying behavioral epochs
MIN_DURATION = 30 # Minimum duration of determined epochs
MIN_GAP = 3 # Minimum gap between epochs so as not to be joined

# settings - dataset details
FS = 1250 # LFP sampling freq
RF = 50 # Running sampling freq

def main():
    # time it
    t_start_script = timer()

    # Define/create directories for outout
    dir_results = f'{PROJECT_PATH}/{RELATIVE_PATH_OUT}'
    if not os.path.exists(f'{dir_results}/npy'): os.makedirs(f'{dir_results}/npy')
    if not os.path.exists(f'{dir_results}/neo'): os.makedirs(f'{dir_results}/neo')
    
    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")

    # get session info for dataset of interest
    sessions_all = cache.get_session_table()
    session_ids = sessions_all[sessions_all['session_type']==SESSION_TYPE].index.values

    # loo[ through all session for dataset of interest
    for i_session, session_id in enumerate(session_ids):
        # display progress
        t_start_session = timer()
        print(f"\n\n Beginning session {i_session+1}/{len(session_ids)}: \t{time_now()}")
        print(f"    session ID: {session_id}")

        # load session data
        session = cache.get_session_data(session_id)

        # get stim info for session - find longest spont epoch and get start/stop time
        stim_table = session.stimulus_presentations
        stim_times = stim_table.sort_values('duration', 
            ascending=False).iloc[0][['start_time','stop_time']].to_numpy()

        # get probe info (for region of interest)
        if REGION is None:
            probe_ids = session.probes.index.values
        else:
            probe_ids, _ = find_probes_in_region(session, REGION)

        # display progress
        print(f"    {len(probe_ids)} probe(s) in ROI")

        # load and epoch behavioral data
        behavior_group = pd.read_pickle(f"{BEHAVIOR_PATH}/{BEHAVIOR_NAME}_{session_id}.pkl")
        behavior_series = behavior_group.analogsignals[BLOCK_POS]

        # Segment behavioral data. NOTE: check that parameters are ok
        above_epochs, below_epochs = get_epoch_times(behavior_series.magnitude.T[0], THRESHOLD, MIN_GAP, MIN_DURATION, RF)
    
        above_epochs += float(behavior_series.t_start)
        below_epochs += float(behavior_series.t_start)
        above_epochs = np.array(split_epochs(above_epochs, MIN_DURATION))
        below_epochs = np.array(split_epochs(below_epochs, MIN_DURATION))
        print(f"Found {len(above_epochs)} above epochs and {len(below_epochs)} below epochs")

            
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
            else:
                chan_ids = session.channels[session.channels.probe_id==probe_id].index.values

            # epoch LFP data for above and below
            above, below = [], []
            time_a, time_b = [], []

            for epoch in above_epochs:
                start_time, end_time = epoch
                lfp_seg = lfp.sel(time = slice(start_time, end_time))
                if lfp_seg.values.shape[0] == FS*MIN_DURATION - 1:
                    lfp_seg = lfp.sel(time = slice(start_time, end_time + 1/FS))
                if lfp_seg.values.shape[0] == FS*MIN_DURATION + 1:
                    lfp_seg = lfp.sel(time = slice(start_time + 1/FS, end_time))
                if lfp_seg.values.shape[0] != FS*MIN_DURATION:
                    print("LFP segment for epoch incorrect dimensions")
                    continue
                above.append(lfp_seg.values)
                time_a.append(np.linspace(start_time, end_time, FS))
                
            if len(above) > 0:
                above = np.stack(above)
            else:
                above = np.array([])
            time_a = np.array(time_a)

            for epoch in below_epochs:
                start_time, end_time = epoch
                lfp_seg = lfp.sel(time = slice(start_time, end_time))
                if lfp_seg.values.shape[0] == FS*MIN_DURATION - 1:
                    lfp_seg = lfp.sel(time = slice(start_time, end_time + 1/FS))
                if lfp_seg.values.shape[0] == FS*MIN_DURATION + 1:
                    lfp_seg = lfp.sel(time = slice(start_time + 1/FS, end_time))
                if lfp_seg.values.shape[0] != FS*MIN_DURATION:
                    print("LFP segment for epoch incorrect dimensions")
                    continue
                below.append(lfp_seg.values)
                time_b.append(np.linspace(start_time, end_time, FS))
                
            if len(below) > 0:
                below = np.stack(below)
            else:
                below = np.array([])
            time_b = np.array(time_b)


            t_start = {'above': [t[0] for t in above_epochs], 
            'below': [t[0] for t in below_epochs]}
            block = create_spont_neo_block(above, below, FS, chan_ids, t_start)

            # add Neo block annotations
            block.annotate(session_type = SESSION_TYPE)
            block.annotate(region = REGION)
            block.annotate(epoch_behavior = BEHAVIOR_NAME)
            block.annotate(stimulus_code = STIM_CODE)
            block.annotate(session_id = session_id)
            block.annotate(probe_id = probe_id)
            block.annotate(stimulus_start_time = stim_times[0]*pq.s)
            block.annotate(stimulus_stop_time = stim_times[1]*pq.s)

            # save results
            print('    saving data')
            fname_out = f"{session_id}_{probe_id}_lfp"
            dir_results = f'{PROJECT_PATH}/{RELATIVE_PATH_OUT}'
            # Only save if epochs exist
            if len(above) > 0:
                np.savez(f"{dir_results}/npy/{fname_out}_above_epochs.npz", 
                lfp=np.swapaxes(np.swapaxes(above, 0, 2), 1, 2), time=time_a) # save lfp array as .npz
            if len(below) > 0:
                np.savez(f"{dir_results}/npy/{fname_out}_below_epochs.npz", 
                    lfp=np.swapaxes(np.swapaxes(below, 0, 2), 1, 2), time=time_b)
            save_pkl(block, f"{dir_results}/neo/{fname_out}_epochs.pkl") # save Neo object as .pkl


        # display progress
        _, min, sec = hour_min_sec(timer() - t_start_session)
        print(f"    session complete in {min} min and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start_script)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def create_spont_neo_block(above, below, fs, chan_id, t_start, block_name=None, units = 'uV'):

    # imports
    from neo.core import Block, Group, Segment, AnalogSignal
    
    # create Neo Block object
    if block_name is None:
        block = Block()
    else:
        block = Block(name=block_name)

    for epoch_type, label in zip([above, below], ['above', 'below']):
        # create Neo Segment for each trial
        epoch_group = Group(name=f'{BEHAVIOR_NAME}_{label}')
        for epoch_idx in range(len(epoch_type)):
            segment = Segment(name=f'trial_{epoch_idx}_{label}')
            block.segments.append(segment)

            # add LFP data
            lfp_as = AnalogSignal(epoch_type[epoch_idx], units=units, sampling_rate=fs*pq.Hz, 
                t_start=t_start[label][epoch_idx]*pq.s)
            lfp_as.annotate(label='lfp', ecephys_channel_id=chan_id)
            segment.analogsignals.append(lfp_as)
            epoch_group.segments.append(segment)
            
        block.groups.append(epoch_group)

    return block

if __name__ == '__main__':
    main()
