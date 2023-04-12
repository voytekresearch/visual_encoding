"""
Combine lfp, spiking and behavioral data into a single Neo Block object for
idiosyncratic spontaneous section.

"""

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
REPO_PATH = "C:/Users/soysa/Documents/Git/visual_encoding"
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'spontaneous' # name for output folder (stimulus of interest)

# spike data regions of interest
REGIONS = ['VISp','LGd']

# settings - dataset details
FS = 1250 # LFP/running sampling freq
# imports
import os
import numpy as np
import pandas as pd
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from time import time as timer
import neo

# custom
import sys
sys.path.append(REPO_PATH) # change this to work from visual_encoding dir
from allen_vc.epoch_extraction_tools import get_epoch_times
from utils import hour_min_sec, save_pkl, get_neo_group_names


def main():
    # time it
    t_start = timer()

    # make sure to group data accordingly
    RUNNING_IN = f"data/behavior/running/{STIM_CODE}"
    SPIKES_IN = "data/spike_data/spike_times"
    LFP_IN = f"data/lfp_data/lfp_epochs/{STIM_CODE}"
    dir_results = f"{PROJECT_PATH}/data/blocks/epoched"
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # loop through all created LFP files
    files = os.listdir(LFP_IN)
    for i_file, fname_in in enumerate(files):

        session_id = fname_in.split("_")[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing file {i_file+1}/{len(files)}")
        print(f"    {fname_in}")

        block = pickle.load(open(f'{LFP_IN}/{fname_in}', 'rb'))

        for region in REGIONS:
            # Load spike trains, and add to all segments
            spiketrains = pd.read_pickle(f"{PROJECT_PATH}/{SPIKES_IN}/{session_id}_{region}")

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
        
                # add each spiketrain to segment after slicing in time
                for spiketrain in spiketrains:
                    spiketrain_seg = spiketrain.time_slice(block.segments[i_seg].t_start, block.segments[i_seg].t_stop)
                    block.segments[i_seg].spiketrains.append(spiketrain_seg)
                        
                    # add spiketrain to region group
                    group_sr.spiketrains.append(spiketrain)

        # Load running AnalogSignal (ONLY WANT LONGEST BLOCK)
        running_group = pd.read_pickle(f"{PROJECT_PATH}/{RUNNING_IN}/running_{session_id}.pkl")

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


        # annotate block
        block.annotate(group_list=get_neo_group_names(block))

        # save results
        fname_out = f"{session_id}_{STIM_CODE}.pkl"
        save_pkl(block, f"{dir_results}/{fname_out}")

         # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")    


if __name__ == "__main__":
    main()
