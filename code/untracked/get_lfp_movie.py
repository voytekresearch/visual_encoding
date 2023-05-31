"""
Epoch LFP during natural movie watching

"""

# imports
import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from time import time as timer
from time import ctime as time_now

# settings
MANIFEST_PATH = "D:/datasets/allen_vc"
PROJECT_PATH = "G:/Shared drives/visual_encoding"
SESSION_TYPE = 'functional_connectivity'
REGION = "VISp"

# dataset details
DURATION = 30
FS = 1250 # LFP sampling freq

# ! FOR TESTING ONLY !
# SESSION_IDS = [766640955]
OVERWRITE = False

def main():
    # time it
    t_start = timer()

    # Define/create directories
    dir_results_0 = f'{PROJECT_PATH}/data/lfp_epochs/natural_movie'
    dir_results_1 = f'{MANIFEST_PATH}/data/lfp_epochs/natural_movie'
    for dir_results in [dir_results_0, dir_results_1]:
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # Create Allensdk cache object
    # print('loading cache...')
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")

    # get session info for dataset of interest
    sessions_all = cache.get_session_table()
    session_ids = sessions_all[sessions_all['session_type']==SESSION_TYPE].index.values

    # loo[ through all session for dataset of interest
    for i_session, session_id in enumerate(session_ids): 
    # for session_id in SESSION_IDS: # ! FOR TESTING ONLY !
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

            # check for results
            if OVERWRITE == False:
                fname_out = f"{session_id}_{probe_id}_lfp_movie.npz"
                if fname_out in os.listdir(dir_results_1): 
                    print(f'    results completed for probe: {probe_id}')
                    continue
            
            # load LFP data
            print(f'    importing LFP data for probe: {probe_id}')
            lfp = session.get_lfp(probe_id)

            # get LFP for ROI
            if ~ (REGION is None):
                # print("getting LFP for ROI...")
                chan_ids = session.channels[(session.channels.probe_id==probe_id) & \
                    (session.channels.ecephys_structure_acronym==REGION)].index.values
                lfp = lfp.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))

            # get stim info for natural movies / shuffled
            for stim_str, stim_name, stim_name_alt in zip(['movie', 'shuffled'], \
                ['natural_movie_one','natural_movie_shuffled'], \
                ['natural_movie_one_more_repeats','natural_movie_one_shuffled']):
                stim_table = session.stimulus_presentations[ \
                    ((session.stimulus_presentations.stimulus_name==stim_name_alt) | \
                        (session.stimulus_presentations.stimulus_name==stim_name)) & \
                    (session.stimulus_presentations.frame==0)]
                
                # epoch LFP
                print(f'    aligning to stim: \t{stim_str}')
                lfp_a, time = align_lfp(lfp, stim_table.start_time.values, 
                    stim_table.index.values, t_window=[0, DURATION], dt=1/FS)

                # save results
                # print('saving...')
                fname_out = f"{session_id}_{probe_id}_lfp_movie_{stim_str}.npz"
                for dir_results in [dir_results_0, dir_results_1]:
                    np.savez(fname_out, lfp=lfp_a, time=time) 

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