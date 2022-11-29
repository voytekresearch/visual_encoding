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

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_OUT = "data/lfp_epochs/natural_movie" # where to save output relative to both paths above

# settings - data of interest
SESSION_TYPE = 'functional_connectivity' # dataset of interest
REGION = "VISp" # brain structure of interest

# settings - stimulus epoch of interest
# STIM can be a list; each will be saved to a separate file.
STIM = ['natural_movie_one_more_repeats','natural_movie_one_shuffled'] 
STIM_CODE = ['movie', 'shuffled'] # alternate stimulus name for output filename
T_WINDOW = [0, 30]  # epoch bounds (sec) [time_before_stim, tiimem_aftfer_stim]

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
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

        # get probe info (for region of interest)
        if REGION is None:
            probe_ids = session.probes.index.values
        else:
            probe_ids, _ = find_probes_in_region(session, REGION)

        # display progress
        print(f"    {len(probe_ids)} probe(s) in ROI")

        # loop through all probes for region of interst
        for probe_id in probe_ids:
            
            # load LFP data
            print(f'    importing LFP data for probe: {probe_id}')
            lfp = session.get_lfp(probe_id)

            # get LFP for ROI
            if ~ (REGION is None):
                chan_ids = session.channels[(session.channels.probe_id==probe_id) & \
                    (session.channels.ecephys_structure_acronym==REGION)].index.values
                lfp = lfp.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))

            # epoch data around stimulus
            for stim_code, stim_name in zip(STIM_CODE, STIM):
                # get stim info 
                stim_table = session.stimulus_presentations[ \
                    session.stimulus_presentations.stimulus_name==stim_name]
                
                # epoch LFP
                print(f'    aligning to stim: \t{stim_name}')
                lfp_a, time = align_lfp(lfp, stim_table.start_time.values, 
                    stim_table.index.values, t_window=[T_WINDOW[0], T_WINDOW[1]], dt=1/FS)

                # save results
                fname_out = f"{session_id}_{probe_id}_lfp_{stim_code}.npz"
                for base_path in [PROJECT_PATH, MANIFEST_PATH]:
                    dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
                    np.savez(f"{dir_results}/{fname_out}", lfp=lfp_a, time=time) 

        # display progress
        _, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    session complete in {min} min and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def find_probes_in_region(session, region):
    probe_ids = session.probes.index.values
    has_region = np.zeros_like(probe_ids).astype(bool)

    for i_probe, probe_id in enumerate(probe_ids):
        regions = session.channels[session.channels.probe_id == probe_id].ecephys_structure_acronym.unique()
        has_region[i_probe] = region in regions

    ids = probe_ids[has_region]
    names = session.probes.description.values[has_region]

    return ids, names


def align_lfp(lfp, t_stim, ids, t_window=[-1,1], dt=0.001):
    trial_window = np.arange(t_window[0], t_window[1], dt)
    time_selection = np.concatenate([trial_window + t for t in t_stim])

    inds = pd.MultiIndex.from_product((ids, trial_window), 
                                    names=('presentation_id', 'time_from_presentation_onset'))

    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name='aligned_lfp')
    ds = ds.assign(time=inds).unstack('time')

    aligned_lfp = ds['aligned_lfp']

    return aligned_lfp, trial_window

def hour_min_sec(duration):
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs


if __name__ == '__main__':
    main()