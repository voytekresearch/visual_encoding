"""
Step 2 (stimulus epochs): create stimulus segments.
This script can be used to segment data arouund stimulus events. This script loads
the ouput of Step 1 (Neo Block objects containing spiking, running, and pupil data
for a single session) and creates trial epochs based on stimulus event times and
windows of interest.

"""

# settings - stimulus epoch of interest
STIM_CODE = 'natural_movie_one_shuffled' # this will be used to name output folders and files
STIM_PARAMS = dict({
    'stimulus_name' : 'natural_movie_shuffled',
    'frame' : 0
    }) # other stim params
T_WINDOW = [0, 30]  # epoch bounds (sec) [time_before_stim, tiimem_aftfer_stim]

# settings - running-speed smoothing and thresholding
SCORE_RUNNING = True # whether to score running behavior
SMOOTH = True # whether to smooth running data
KERNEL_SIZE = 5 # (s) kernel size for median filter 
THRESHOLD = 1 # (cm/s) minimum for running classification 

# Imports - general
import numpy as np
import os
from time import time as timer
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import quantities as pq
from scipy.ndimage import median_filter

# Imports - custom
import sys
sys.path.append('allen_vc')
from paths import PATH_EXTERNAL
from utils import hour_min_sec
from neo_utils import get_analogsignal
print('Imports complete...')

def main():

    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PATH_EXTERNAL}/data/blocks/segmented/{STIM_CODE}" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load Allen project cache
    manifest_path = f"{PATH_EXTERNAL}/dataset/manifest.json"
    if os.path.exists(manifest_path):
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        print('Project cache loaded...')
    # stop execution if manifest file not found
    else:
        print('Manifest file not found. Please check MANIFEST_PATH.')
        return   
    
    # loop through all files
    dir_input =  f"{PATH_EXTERNAL}/data/blocks/sessions"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # get stim info for session
        session = cache.get_session_data(int(session_id))
        stim_table = session.stimulus_presentations
        for param_name in STIM_PARAMS.keys():
            stim_table = stim_table[stim_table[param_name] == STIM_PARAMS[param_name]]
        stim_times = stim_table.start_time.values

        # load session data
        block = neo.io.NeoMatlabIO(filename=f"{dir_input}/{fname}").read_block() # load Step 1 results
        session_seg = block.segments[0] # unpack session data
        annotations = block.annotations # unpack annotations

        # initialize new block to hold segments and add annotations
        for key, val in STIM_PARAMS.items():
            annotations[key] = val
        block = neo.Block(**annotations)

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
            annotations = {'index' : i_seg, 'stimulus_onset' : t_stim, 'time_window' : T_WINDOW}
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

        # score behavior
        if SCORE_RUNNING:
            block = score_running(block)

        # save results
        fname_out = f"{dir_results}/{fname}"
        neo.io.NeoMatlabIO(fname_out).write_block(block)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def score_running(block):
    # get running data
    speed_raw, _ = get_analogsignal(block, 'running_speed')
    temp = get_analogsignal(block, 'running_speed', segment_idx=0, return_numpy=False)
    fs = float(temp.sampling_rate)

    # smooth running data, if desired
    if SMOOTH:
        speed = np.zeros_like(speed_raw)
        filter_size = int(KERNEL_SIZE * fs)
        for i in range(speed_raw.shape[0]): # filter each row
            speed[i, :] = median_filter(speed_raw[i, :], size=filter_size)
    else:
        speed = speed_raw

    # score running
    running = (speed>THRESHOLD).any(axis=1)

    # add results to block and segment annotations
    block.annotations['running'] = running
    for segment in block.segments:
        segment.annotations['running'] = running[segment.index]

    return block

if __name__ == '__main__':
    main()