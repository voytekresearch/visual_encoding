"""
Step 3: add LFP to stimulus segements.
This script loads the output of Step 2 (segmented Neo Block) and adds the corresponding
LFP data to each segment. 

"""

# settings - directories
STIM_CODES = ['natural_movie_one_shuffled', 
              'spontaneous_running', 'sponteneous_stationary'] # list of results folders

# settings 
BRAIN_STRUCTURE = 'VISp' # regions of interest for LFP data

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

# Imports - custom
import sys
sys.path.append('allen_vc')
from paths import PATH_EXTERNAL
from utils import hour_min_sec
from allen_utils import find_probes_in_region, align_lfp
print('Imports complete...')


def main():
    # time it
    t_start = timer()

    # load Allen project cache
    manifest_path = f"{PATH_EXTERNAL}/dataset/manifest.json"
    if os.path.exists(manifest_path):
        cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        print('Project cache loaded...')
    # stop execution if manifest file not found
    else:
        print('Manifest file not found. Please check MANIFEST_PATH.')
        return        
    
    for i_stim, stim_code in enumerate(STIM_CODES):
        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing stimulus: {stim_code} (stimulus {i_stim+1}/{len(STIM_CODES)})")
        
        # Define/create directories for inputs/outputs
        dir_results = f"{PATH_EXTERNAL}/data/blocks/lfp/{stim_code}"
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)

        # loop through all files
        dir_input =  f"{PATH_EXTERNAL}/data/blocks/segmented/{stim_code}"
        files = os.listdir(dir_input)
        for i_file, fname in enumerate(files):
            session_id = fname.split('_')[1].split('.')[0]

            # display progress
            t_start_f = timer()
            print(f"\n    Analyzing session: {session_id} (stimulus {i_stim+1}/{len(STIM_CODES)}; file {i_file+1}/{len(files)})")

            # load segmented Neo Block for session (Step 2 results)
            block = neo.io.NeoMatlabIO(f"{dir_input}/{fname}").read_block()

            # skip file if no segments found
            if block.segments == []:
                print(f"        No segments found... skipping")
                continue

            # load session data (for LFP dataset)
            session = cache.get_session_data(int(session_id))

            # get probe info (for region(s) of interest)
            if BRAIN_STRUCTURE is None:
                probe_ids = session.probes.index.values
                print(f"        {len(probe_ids)} probe(s) identified")
            else:
                probe_ids, _ = find_probes_in_region(session, BRAIN_STRUCTURE)
                print(f"        {len(probe_ids)} probe(s) in ROI")

            # annotate block
            block.annotate(lfp_brain_structure=BRAIN_STRUCTURE, lfp_probe_ids=probe_ids, 
                            has_lfp_data=True)

            # loop through all probes for region of interst
            for probe_id in probe_ids:
                # skip probes with no LFP data
                if ~ session.probes.loc[probe_id, 'has_lfp_data']:
                    print(f"        No LFP data for probe: {probe_id}... skipping")
                    continue

                # import LFP data
                print(f'        importing LFP data for probe: {probe_id}')
                lfp = session.get_lfp(probe_id)

                # get LFP for ROI
                if BRAIN_STRUCTURE is not None:
                    chan_ids = session.channels[(session.channels.probe_id==probe_id) & \
                        (session.channels.ecephys_structure_acronym==BRAIN_STRUCTURE)].index.values
                    lfp = lfp.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
                else:
                    chan_ids = session.channels[session.channels.probe_id==probe_id].index.values

                # align lfp to existing segments
                start_times = []
                for segment in block.segments:
                    start_times.append(float(segment.t_start))
                duration = float(block.segments[0].t_stop - block.segments[0].t_start)
                lfp_a, _ = align_lfp(lfp, start_times, t_window=[0, duration], dt=1/FS)
                
                # prepare annotations
                chan_ids = np.intersect1d(chan_ids, lfp.channel.values) # remove channels with no LFP data
                annotations = {'data_type' : 'lfp', 'probe_id': probe_id, 'brain_structure': BRAIN_STRUCTURE, 
                            'channel_ids': chan_ids}
                if len(probe_ids) == 1:
                    lfp_name = "lfp"
                else:
                    lfp_name = f"lfp_{probe_id}"

                # create group for probe
                group = neo.Group(name=f"lfp_{probe_id}")

                # loop through segments and add LFP data
                for i_seg, segment in enumerate(block.segments):
                    # create neo analogsignal for segment
                    lfp_segment = neo.AnalogSignal(lfp_a[i_seg].T, units=UNITS, sampling_rate=FS*pq.Hz, 
                                                name=lfp_name, t_start=segment.t_start, **annotations)

                    # add LFP data to segment and group
                    segment.analogsignals.append(lfp_segment)
                    group.analogsignals.append(lfp_segment)

                # add group to block
                block.groups.append(group)

            # save results
            print(f"        saving results...")
            neo.io.NeoMatlabIO(filename=f"{dir_results}/{fname}").write_block(block)

            # display progress
            hour, min, sec = hour_min_sec(timer() - t_start_f)
            print(f"        file complete in {hour} hour, {min} min, and {sec :0.1f} s")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"\n    {stim_code} completed in: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()