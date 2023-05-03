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
STIM_CODE = 'natural_movie_one_shuffled' # this will be used to identify input/output folders

# settings - regions of interest for LFP data
BRAIN_STRUCTURE = 'VISp'

# settings - dataset details
FS = 1250 # LFP sampling freq
UNITS = 'uV' # LFP units

# Imports - general
import numpy as np
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
from allen_utils import find_probes_in_region
print('Imports complete...')


def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PROJECT_PATH}/data/blocks_lfp/{STIM_CODE}"
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load Allen project cache
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")
    print('Project cache loaded...')
    
    # loop through all files
    dir_input =  f"{PROJECT_PATH}/data/blocks_segmented/{STIM_CODE}"
    files = os.listdir(dir_input)
    for i_file, fname in enumerate(files):
        session_id = fname.split('_')[1].split('.')[0]

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {session_id} ({i_file+1}/{len(files)})")

        # load segmented Neo Block for session (Step 2 results)
        block = pd.read_pickle(f"{dir_input}/{fname}")

        # load session data (for LFP dataset)
        session = cache.get_session_data(int(session_id))

        # get probe info (for region(s) of interest)
        if BRAIN_STRUCTURE is None:
            probe_ids = session.probes.index.values
            print(f"    {len(probe_ids)} probe(s) identified")
        else:
            probe_ids, _ = find_probes_in_region(session, BRAIN_STRUCTURE)
            print(f"    {len(probe_ids)} probe(s) in ROI")

        # loop through all probes for region of interst
        for probe_id in probe_ids:
            # skip probes with no LFP data
            if ~ session.probes.loc[probe_id, 'has_lfp_data']:
                print(f"    No LFP data for probe: {probe_id}... skipping")
                continue

            # import LFP data
            print(f'    importing LFP data for probe: {probe_id}')
            lfp = session.get_lfp(probe_id)

            # get LFP for ROI
            if BRAIN_STRUCTURE is not None:
                chan_ids = session.channels[(session.channels.probe_id==probe_id) & \
                    (session.channels.ecephys_structure_acronym==BRAIN_STRUCTURE)].index.values
                lfp = lfp.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
            else:
                chan_ids = session.channels[session.channels.probe_id==probe_id].index.values

            # create Neo AnalogSignal for LFP data for the whole session
            annotations = {'data_type' : 'lfp', 'probe_id': probe_id, 'brain_structure': BRAIN_STRUCTURE, 
                           'channel_ids': chan_ids}
            if len(probe_ids) == 1:
                lfp_name = "lfp"
            else:
                lfp_name = f"lfp_{probe_id}"
            lfp = neo.AnalogSignal(lfp, units=UNITS, sampling_rate=FS*pq.Hz, name=lfp_name, **annotations)

            # create group for probe
            group = neo.Group(name=f"lfp_{probe_id}")

            # loop through segments and add LFP data
            for segment in block.segments:
                # slice LFP data according to segment start/end times
                lfp_segment = lfp.time_slice(segment.t_start, segment.t_stop)

                # add LFP data to segment and group
                segment.analogsignals.append(lfp_segment)
                group.analogsignals.append(lfp_segment)

            # add group to block
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