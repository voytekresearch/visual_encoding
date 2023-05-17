"""
Step 1: create stimulus segments.
This script can be used to segment data arouund stimulus events. This script loads
the ouput of Step 1 (Neo Block objects containing spiking, running, and pupil data
for a single session) and creates trial epochs based on stimulus event times and
windows of interest.

"""

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory

# settings - smoothing
SMOOTH = True
KERNEL_SIZE = 5 # kernel size for median filter (s)

# settings - epoch extraction
THRESHOLD = 0.5
MIN_GAP = 1 # minimum gap between epochs (s)
MIN_DURATION = 30 # minimum duration of epochs (s)

# dataset details
FS = 50 # running sampling freq

# Imports - general
import os
from time import time as timer
import neo
import quantities as pq
import pandas as pd
from scipy.ndimage import median_filter

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import hour_min_sec
from epoch_extraction_tools import get_epoch_times, split_epochs
from neo_utils import get_analogsignal
print('Imports complete...')

def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PROJECT_PATH}/data/blocks/segmented/spontaneous"
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # loop through all files
    dir_input =  f"{PROJECT_PATH}/data/blocks/spontaneous"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # load session data
        block = neo.io.NeoMatlabIO(filename=f"{dir_input}/{fname}").read_block() # load Step 1 results
        session_seg = block.segments[0] # unpack session data
        annotations = block.annotations # unpack annotations

        # get running data
        speed, time = get_analogsignal(block, 'running_speed')

        # smooth running data
        if SMOOTH:
            speed_smooth = median_filter(speed, size=int(KERNEL_SIZE*FS))

        # find segments
        epochs_r, epochs_s = get_epoch_times(speed_smooth, threshold=THRESHOLD, min_gap=MIN_GAP, min_duration=MIN_DURATION, fs=FS)
        epochs_r = split_epochs(epochs_r, MIN_DURATION)
        epochs_s = split_epochs(epochs_s, MIN_DURATION)
        epochs_r = epochs_r + time[0]
        epochs_s = epochs_s + time[0]

        # create Neo Segments based on epoch times for running and stationary
        for behavior, epochs in zip(['running', 'stationary'], [epochs_r, epochs_s]):

            # initialize new block to hold segments and add annotations
            annotations['behavior'] = behavior
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
            for t_start, t_stop in epochs:
                # define time window of interest
                t_seg = [t_start, t_stop]*pq.s

                # create segment
                segment = neo.Segment()

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
            fname_out = f"block_{session_id}_{behavior}.mat"
            neo.io.NeoMatlabIO(f"{dir_results}/{fname_out}").write_block(block)

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()