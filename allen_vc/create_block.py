"""
Combine lfp, spiking and behavioral data into a single Neo Block object.

"""

# imports
import os
import numpy as np
import pandas as pd
import pickle
from time import time as timer
from utils import hour_min_sec, save_pkl
from neo.core import Group

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
RELATIVE_PATH_LFP = "data/lfp_data/lfp_epochs_neo/natural_movie/pkl" # folder containing output of epoch_lfp.py
RELATIVE_PATH_SPIKES = "data/spike_data/spike_times" # folder containing output of spike_data()
RELATIVE_PATH_RUNNING = "data/behavior/running" # folder containing output of comp_velocity_for_block.py
RELATIVE_PATH_OUT = "data/blocks" # where to save output relative to both paths above
STIMULUS_NAME = "natural_movie_one_more_repeats"

# spike data regions of interest
REGIONS = ['VISp','LGd']

# settings - dataset details
FS = 1250 # LFP sampling freq

def main():
    # time it
    t_start = timer()

    # Define/create directories for outout
    for base_path in [PROJECT_PATH]:#, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # id files of interst and loop through them
    files = os.listdir(f'{PROJECT_PATH}/{RELATIVE_PATH_LFP}')
    for i_file, fname_in in enumerate(files):

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        # load Neo Block containing LFP
        block = pickle.load(open(f"{PROJECT_PATH}/{RELATIVE_PATH_LFP}/{fname_in}", "rb"))

        # load spike data and add to block
        for region in REGIONS:

            # Don't add if brain region not recorded in session
            fname_spikes = f"{fname_in.split('_')[0]}_{region}.pkl"
            if not os.path.exists(f"{PROJECT_PATH}/{RELATIVE_PATH_SPIKES}/{fname_spikes}"):
                continue

            # load list of spiketrains for region
            spiketrains = pd.read_pickle(f"{PROJECT_PATH}/{RELATIVE_PATH_SPIKES}/{fname_spikes}")

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
                    group_sr.spiketrains.append(spiketrain)



        # Still working on this part/needs to be bug fixed!

        # load behavioral data
        fname_running = f"running_{fname_in.split('_')[0]}.pkl"
        running_group = pd.read_pickle(f"{PROJECT_PATH}/{RELATIVE_PATH_RUNNING}/{STIMULUS_NAME}/{fname_running}")

        # loop through segments
        for i_seg in range(len(block.segments)):
            # filter for correct signal based on start/stop times
            for signal in running_group.analogsignals:
                if (signal.t_start<=block.segments[i_seg].t_start) and \
                signal.t_stop>=block.segments[i_seg].t_stop:
                    running_series = signal
                    found = True

            if not found:
                print(f"Did not find series for {i_seg}th segment!!!")

            # slice signal and append to segment
            running_seg = running_series.time_slice(block.segments[i_seg].t_start, block.segments[i_seg].t_stop)
            block.segments[i_seg].analogsignals.append(running_seg)

            # Testing
            found = False


        # save results
        fname_out = f"{fname_in.split('_')[0]}_{STIMULUS_NAME}.pkl"
        for base_path in [PROJECT_PATH]:#, MANIFEST_PATH]:
            dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
            save_pkl(block, f"{dir_results}/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()