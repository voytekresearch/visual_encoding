"""
Step 1 (Spontaneous pipeline): create spontaneous block.
Create a Neo Block object for the longest spontaneous epoch in each session.

"""

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory

# Imports - general
import os
from time import time as timer
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import quantities as pq

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import hour_min_sec
print('Imports complete...')


def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PROJECT_PATH}/data/blocks/spontaneous" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load Allen project cache
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")
    print('Project cache loaded...')
    
    # loop through all files
    dir_input =  f"{PROJECT_PATH}/data/blocks/sessions"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # get stim info for session and find longest spontaneous epoch
        session = cache.get_session_data(int(session_id))
        stim_table = session.stimulus_presentations
        st_spont = stim_table[stim_table['stimulus_name'] == 'spontaneous'].reset_index()
        start_time = st_spont.loc[st_spont['duration'].argmax(), 'start_time']
        stop_time = st_spont.loc[st_spont['duration'].argmax(), 'stop_time']
        t_seg = [start_time, stop_time]*pq.s # time window of interest

        # load session data and reset block
        block = neo.io.NeoMatlabIO(filename=f"{dir_input}/{fname}").read_block() # load Step 1 results
        session_seg = block.segments[0] # unpack session data
        block.segments.pop(0) # drop segment
        segment = neo.Segment(name="spontaneous") # create new segment

        # add each spiketrain to segment after slicing in time
        for spiketrain in session_seg.spiketrains:
            spiketrain_seg = spiketrain.time_slice(*t_seg)
            segment.spiketrains.append(spiketrain_seg)

        # add running wheel and pupil tracking data to segments
        for signal in session_seg.analogsignals:
            signal_seg = signal.time_slice(*t_seg)
            segment.analogsignals.append(signal_seg)

        # add segment to block
        block.segments.append(segment)

        # save results
        neo.io.NeoMatlabIO(f"{dir_results}/{fname}").write_block(block)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()