"""
Combine lfp, spiking and behavioral data into a single Neo Block object for
idiosyncratic spontaneous section.

"""

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
REPO_PATH = "C:/Users/soysa/Documents/Git/visual_encoding"
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
STIM_CODE = 'spontaneous' # name for output folder (stimulus of interest)
SESSION_TYPE = 'functional_connectivity'
RUNNING_IN = f"{PROJECT_PATH}/data/lfp_data/lfp_epochs/spont"

# spike data regions of interest
REGIONS = ['VISp','LGd']

# settings - dataset details
FS = 1250 # LFP sampling freq
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


# NOTE: debugging is needed and this is just a start!


def main():
	# Outline:
	# Instantiate a block for each session
	# Load/segment running series and create segments
	# Load/add lfp data
	# Add proper spiking data


	# loop through all created LFP files
	for i_file, fname_in in enumerate(os.listdir(RUNNING_IN)):

		session_id = fname_in.split("_")[0]
		print(f"Analyzing Session: \t{session_id}")

		# Load running AnalogSignal
		running_series = pd.read_pickle(f"{PROJECT_PATH}/data/behavior/running/{STIM_CODE}/running_{session_id}.pkl")

		block = init_spont_blocks(running_series, FS, ['running', 'stationary'])

		# Load lfp array, create AnalogSignal and add to all segments
		lfp_series = np.load(fname_in)
		lfp_sig = AnalogSignal(lfp_series, units="uV", sampling_rate=FS*pq.Hz, 
            t_start=lfp_series[0])
        lfp_sig.annotate(label='lfp', ecephys_channel_id=fname_in.split("_")[1])

		for i_seg, seg in enumerate(block.segments):
			sliced = lfp_seg.time_slice(seg.t_start, seg.t_stop)
			seg.analogsignals.append(sliced)

	

def init_spont_blocks(series, fs, above_below_names=None, block_name=None):

	if block_name is None:
        block = Block()
    else:
        block = Block(name=block_name)

    # Segment behavioral data
    above_epochs, below_epochs = get_epoch_times(series.signal, THRESHOLD, MIN_DURATION)

    def add_segments(block, epochs, name=None):
    	# Create Neo segments based on positive/negative behavioral epochs and add to block

    	for i, epoch in enumerate(epochs):
    		if name is not None:
    			segment = neo.core.Segment(name=f'{name}_{i}')
    		else:
    			segment = neo.core.Segment(name=f'trial_{i}')
	        block.segments.append(segment)

	        # add LFP data
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