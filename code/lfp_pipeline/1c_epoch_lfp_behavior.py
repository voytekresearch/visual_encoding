"""
Epoch LFP during longest spontaneous epoch according to running/stationary behavior

"""
# Set paths
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_OUT = "data/lfp_data/lfp_epochs/spont/running" # where to save output relative to both paths above 
# (will have both running and pupil folders ideally)

# imports
import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from time import time as timer
from time import ctime as time_now
from utils import find_probes_in_region, hour_min_sec
from epoch_extraction_tools import get_epoch_times
from 1_epoch_lfp import create_neo_block

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest
REGION = "VISp" # brain structure of interest
BEHAVIOR_PATH = "C:/.."

# settings - stimulus epoch of interest
STIM_PARAMS = dict({'stimulus_name' : 'spontaneous'})
# DURATIONS = [1, 30]  # duration of arbitrary epochs (s)
STIM_CODE = 'spont' # alternate stimulus name for output filename
THRESHOLD = 1 # Threshold for identifying behavioral epochs
MIN_DURATION = 1 # Minimum duration of determined epochs
MIN_GAP = 0.1 # Minimum gap between epochs so as not to be joined

# settings - dataset details
FS = 1250 # LFP sampling freq
# EXPECTED_DURATION = 30*60 # duration of spontaneous epoch (s)

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    dir_results = f'{PROJECT_PATH}/{RELATIVE_PATH_OUT}'
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=f"{PROJECT_PATH}/dataset/manifest.json")

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

        # get stim info for session - find longest spont epoch
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

            # load and epoch behavioral data
            running_group = pd.read_pickle(f"{BEHAVIOR_PATH}")
            running_series = running_group.analogsignals[4] # this is only the case for running, make general

            # Segment behavioral data. NOTE: check that parameters are ok
            above_epochs, below_epochs = get_epoch_times(series.signal, THRESHOLD, MIN_GAP, MIN_DURATION, FS) # Running FS and LFP FS the same?

            # epoch LFP data for above and below
            above, below = [], []
            for epoch in above_epochs:
                start_time, end_time = epoch
                lfp_seg = lfp.sel(time = slice(start_time, end_time))
                above.append(lfp_seg.to_numpy())

            for epoch in below_epochs:
                start_time, end_time = epoch
                lfp_seg = lfp.sel(time = slice(start_time, end_time))
                below.append(lfp_seg.to_numpy())

            # how should channels be selected/aggregated here?

            # save results
            print('    saving data')
            fname_out = f"{session_id}_{probe_id}_lfp_{chan_ids}.npz" # channel id?
            dir_results = f'{PROJECT_PATH}/{RELATIVE_PATH_OUT}'
            np.savez(f"{dir_results}/{fname_out}", above=above, below=below)


        # display progress
        _, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    session complete in {min} min and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")



if __name__ == '__main__':
    main()
