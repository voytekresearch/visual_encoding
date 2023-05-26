"""
Combine lfp, spiking and behavioral data into a single Neo Block object.

"""

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'spontaneous' # name for output folder (stimulus of interest)
LFP_BLOCKS = f"{PROJECT_PATH}/data/lfp_data/lfp_epochs/spontaneous/running/neo/"

# spike data regions of interest
REGIONS = ['VISp','LGd']

# settings - annotations
RUNNING_ANNOTATIONS = False

# imports
import os
import numpy as np
import pandas as pd
import pickle
from time import time as timer
from neo.core import Group

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import hour_min_sec, save_pkl, get_neo_group_names

def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    path_spikes = f"{PROJECT_PATH}/data/spike_data/spike_times/"
    dir_results = f"{PROJECT_PATH}/data/blocks" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)
    
    # id files of interst and loop through them
    files = os.listdir(LFP_BLOCKS)
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        # load Neo Block containing LFP
        block = pickle.load(open(f"{LFP_BLOCKS}/{fname_in}", "rb"))

        # load spike data and add to block
        for region in REGIONS:

            # Don't add if brain region not recorded in session
            fname_spikes = f"{fname_in.split('_')[0]}_{region}.pkl"
            if not os.path.exists(f"{path_spikes}/{fname_spikes}"):
                continue

            # load list of spiketrains for region
            spiketrains = pd.read_pickle(f"{path_spikes}/{fname_spikes}")

            # create Group object for each unit
            for spiketrain in spiketrains:
                group_name = f"unit_{spiketrain.annotations['unit_id']}"
                group_u = Group([spiketrain], name=group_name)
                block.groups.append(group_u)

            # loop through segments
            for i_seg in range(len(block.segments)):
                # create Group object for each segment-region combination
                group_sr_name = f"seg_{i_seg}_{region}"
                group_sr = Group(name=group_sr_name)
                block.groups.append(group_sr)

                # add LFP segment to group
                group_sr.analogsignals.append(block.segments[i_seg].analogsignals)

                # add each spiketrain to segment after slicing in time
                for spiketrain in spiketrains:
                    spiketrain_seg = spiketrain.time_slice(block.segments[i_seg].t_start, block.segments[i_seg].t_stop)
                    block.segments[i_seg].spiketrains.append(spiketrain_seg)
                    
                    # add spiketrain to region group
                    group_sr.spiketrains.append(spiketrain_seg)

        # load behavioral data
        fname_running = f"running_{fname_in.split('_')[0]}.pkl"
        running_group = pd.read_pickle(f"{PROJECT_PATH}/data/behavior/running/{STIM_CODE}/{fname_running}")

        # initialize array for running annotation
        running = np.zeros(len(block.segments))

        # loop through segments
        for i_seg in range(len(block.segments)):
            # filter for correct signal based on start/stop times
            for signal in running_group.analogsignals:
                if (signal.t_start<=block.segments[i_seg].t_start) and \
                signal.t_stop>=block.segments[i_seg].t_stop:
                    running_series = signal
                    break

            # slice signal and append to segment
            running_seg = running_series.time_slice(block.segments[i_seg].t_start, block.segments[i_seg].t_stop)
            block.segments[i_seg].analogsignals.append(running_seg)
            block.segments[i_seg].analogsignals[-1].name = 'running_speed'

            # determine if running during segment then annotate segment
            if RUNNING_ANNOTATIONS:
                speed = running_seg.magnitude
                running[i_seg] = np.any(speed > 1)
                block.segments[i_seg].annotations['running'] = running[i_seg].astype(bool)

        # save running boolean array to block
        if RUNNING_ANNOTATIONS:
            block.annotations['running'] = running.astype(bool)

        # annotate block
        block.annotate(group_list=get_neo_group_names(block))
        block.annotate(analogsignals=['lfp', 'running_speed', 'pupil_area'])

        # save results
        fname_out = f"{fname_in.split('_')[0]}_{STIM_CODE}.pkl"
        save_pkl(block, f"{dir_results}/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()