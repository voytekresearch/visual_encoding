"""
Step 2: create stimulus segments.
This script can be used to segment data arouund stimulus events. This script loads
the ouput of Step 1 (Neo Block objects containing spiking, running, and pupil data
for a single session) and creates trial epochs based on stimulus event times and
windows of interest.

"""

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'natural_movie_one_shuffled' # this will be used to name output folders and files

# settings - stimulus epoch of interest
STIM_PARAMS = dict({
    'stimulus_name' : 'natural_movie_one_shuffled',
    'frame' : 0
    }) # other stim params
T_WINDOW = [0, 30]  # epoch bounds (sec) [time_before_stim, tiimem_aftfer_stim]

# Imports - general
import os
from time import time as timer
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import quantities as pq
import pandas as pd

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import hour_min_sec, save_pkl
print('Imports complete...')


def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PROJECT_PATH}/data/blocks_segmented/{STIM_CODE}" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load Allen project cache
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")
    print('Project cache loaded...')
    
    # loop through all files
    dir_input =  f"{PROJECT_PATH}/data/blocks_session"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # BUG: this files produces an error
        if ((session_id == '768515987') | (session_id == '840012044')):
            print('    Skipping...')
            continue

        # get stim info for session
        session = cache.get_session_data(int(session_id))
        stim_table = session.stimulus_presentations
        for param_name in STIM_PARAMS.keys():
            stim_table = stim_table[stim_table[param_name] == STIM_PARAMS[param_name]]
        stim_times = stim_table.start_time.values

        # load session data and initialize new block to hold segments
        block = pd.read_pickle(f"{dir_input}/{fname}") # load Step 1 results
        session_seg = block.segments[0] # unpack session data
        block = neo.Block() # init new block

        # create Neo Group for each unit
        for spiketrain in session_seg.spiketrains:
            group = neo.Group(name=f"unit_{spiketrain.name}")
            block.groups.append(group)
        
        # create Neo Group for eah analog signal type
        signal_groups = dict()
        signal_groups['running_speed'] = neo.Group(name='running_speed')
        signal_groups['pupil_area'] = neo.Group(name='pupil_area')

        # create Neo Semgments based on stimulus times
        for i_seg, t_stim in enumerate(stim_times):
            # define time window of interest
            t_seg = [t_stim+T_WINDOW[0], t_stim+T_WINDOW[1]]*pq.s

            # create segment and add annotations
            annotations = {'index' : i_seg, 'stimulus_onset' : t_stim, 'time_window' : T_WINDOW,
                            'stimulus_parameters' : STIM_PARAMS}
            segment = neo.Segment(**annotations)

            # add each spiketrain to segment after slicing in time
            for i_unit, spiketrain in enumerate(session_seg.spiketrains):
                spiketrain_seg = spiketrain.time_slice(*t_seg)
                segment.spiketrains.append(spiketrain_seg)
                block.groups[i_unit].spiketrains.append(spiketrain_seg)

            # add running wheel and pupil tracking data to segments
            for a_signal in session_seg.analogsignals:
                signal = a_signal.time_slice(*t_seg)
                segment.analogsignals.append(signal)
                signal_groups[signal.name].analogsignals.append(signal)

            # add segment to block
            block.segments.append(segment)

        # add signal groups to block
        for group in signal_groups.values():
            block.groups.append(group)

        # save results
        save_pkl(block, f"{dir_results}/{fname}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()