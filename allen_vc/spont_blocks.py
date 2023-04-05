"""
Combine lfp, spiking and behavioral data into a single Neo Block object for
idiosyncratic spontaneous section.

"""

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
REPO_PATH = "C:/Users/soysa/Documents/Git/visual_encoding"
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'spont' # name for output folder (stimulus of interest)
SESSION_TYPE = 'functional_connectivity'
RUNNING_IN = f"{PROJECT_PATH}/data/lfp_data/lfp_epochs/spont"
THRESHOLD = 1 # Threshold for identifying behavioral epochs
MIN_DURATION = 1 # Minimum duration of determined epochs

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
sys.path.append(REPO_PATH)
from allen_vc.epoch_extraction_tools import get_epoch_times
from utils import hour_min_sec, save_pkl, get_neo_group_names


# NOTE: debugging is needed and this is just a start!


def main():

    # make sure to group data accordingly
    RUNNING_IN = f"data/behavior/running/{STIM_CODE}"
    SPIKES_IN = "data/spike_data/spike_times"
    LFP_IN = f"data/lfp_data/lfp_epochs/{STIM_CODE}"
    dir_results = f"{PROJECT_PATH}/data/blocks"
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # loop through all created LFP files
    for i_file, fname_in in enumerate(os.listdir(LFP_IN)):

        session_id = fname_in.split("_")[0]
        print(f"Analyzing Session: \t{session_id}")

        # Load running AnalogSignal (ONLY WANT LONGEST BLOCK)
        running_group = pd.read_pickle(f"{PROJECT_PATH}/{RUNNING_IN}/running_{session_id}.pkl")
        running_series = running_group.analogsignals[4] # make sure proper index

        block = init_spont_blocks(running_series, FS, ['running', 'stationary'])

        # Load lfp array, create AnalogSignal and add to all segments
        lfp_series = np.load(fname_in)
        lfp_sig = AnalogSignal(lfp_series, units="uV", sampling_rate=FS*pq.Hz, 
            t_start=lfp_series[0])
        lfp_sig.annotate(label='lfp', ecephys_channel_id=fname_in.split("_")[1])

        for i_seg, seg in enumerate(block.segments):
            sliced = lfp_seg.time_slice(seg.t_start, seg.t_stop)
            seg.analogsignals.append(sliced)

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

                # add LFP segment to group
                group_sr.analogsignals.append(block.segments[i_seg].analogsignals[0]) # not sure 0 or 1

                # add each spiketrain to segment after slicing in time
                for spiketrain in spiketrains:
                    spiketrain_seg = spiketrain.time_slice(block.segments[i_seg].t_start, block.segments[i_seg].t_stop)
                    block.segments[i_seg].spiketrains.append(spiketrain_seg)
                        
                    # add spiketrain to region group
                    group_sr.spiketrains.append(spiketrain)

        # annotate block
        block.annotate(group_list=get_neo_group_names(block))

        # More annotations?

        # save results
        fname_out = f"{session_id}_{STIM_CODE}.pkl"
        save_pkl(block, f"{dir_results}/{fname_out}")
    



def init_spont_blocks(series, fs, above_below_names=None, block_name=None):

    if block_name is None:
        block = Block()
    else:
        block = Block(name=block_name)

    # Segment behavioral data. NOTE: need to make sure signal is handled properly and proper timescale
    above_epochs, below_epochs = get_epoch_times(series.signal, THRESHOLD, MIN_DURATION)

    def add_segments(block, epochs, name=None):
        # Create Neo segments based on positive/negative behavioral epochs and add to block

        for i, epoch in enumerate(epochs):
            if name is not None:
                segment = neo.core.Segment(name=f'{name}_{i}')
            else:
                segment = neo.core.Segment(name=f'trial_{i}')
            block.segments.append(segment)

            # add behavioral signal data
            sliced = series.time_slice(epoch[0], epoch[1])
            segment.analogsignals.append(sliced)

    if above_below_names is not None:
        add_segments(block, above_epochs, above_below_names[0])
        add_segments(block, below_epochs, above_below_names[1])
    else:
        add_segments(block, above_epochs)
        add_segments(block, below_epochs)

    return block




if __name__ == "__main__":
    main()