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
DIR_EPOCH_DATA = 'data/behavioral_epochs' # folder containg behavioral epoch times

# Imports - general
import numpy as np
import os
from time import time as timer
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
    dir_results = f"{PROJECT_PATH}/data/blocks/spontaneous" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # loop through all files
    dir_input =  f"{PROJECT_PATH}/data/session_blocks"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # load session data and initialize new block to hold segments
        block = pd.read_pickle(fname) # load Step 1 results
        session_seg = block.segments[0] # unpack session data
        block = neo.Block() # init new block

        # load behavioral epoch times for session
        epoch_data = np.load(f"{DIR_EPOCH_DATA}/{session_id}.npz")
        
        # loop through behaviors
        for behavior in epoch_data.keys():
            # display progress
            print(f"  {behavior}...")

            # create Neo Semgments based on stimulus times
            for i_seg, t_start, t_stop in enumerate(epoch_data[behavior]):
                    # define time window of interest
                    t_seg = [t_start*pq.s, t_stop*pq.s]

                    # create segment and add annotations
                    annotations = {'index' : i_seg, 'start_time' : t_start, 
                                   'stop_time' : t_stop, 'behavior' : behavior}
                    segment = neo.Segment(**annotations)

                    # add each spiketrain to segment after slicing in time
                    for spiketrain in session_seg.spiketrains:
                        spiketrain_seg = spiketrain.time_slice(*t_seg)
                        segment.spiketrains.append(spiketrain_seg)

                    # add running wheel and pupil tracking data to segments
                    for a_signal in session_seg.analogsignals:
                        segment.analogsignals.append(a_signal.time_slice(*t_seg))

                    # add segment to block
                    block.segments.append(segment)

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