"""
Combine lfp, spiking and behavioral data into a single Neo Block object.

"""

# settings - directories
MANIFEST_PATH = "D:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory

# settings - regions of interest for spike data
BRAIN_STRUCTURES = ['VISp','LGd']

# settings - sampling frequency
FS_PUPIL = 50
FS_RUNNING = 50

# Imports - general
import os
from time import time as timer
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import quantities as pq

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import hour_min_sec, save_pkl
from allen_utils import create_neo_spiketrains, compute_running_speed, compute_pupil_area


def main():
    # time it
    t_start = timer()

    # Define/create directories for inputs/outputs
    dir_results = f"{PROJECT_PATH}/data/session_blocks" 
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load Allen project cache
    cache = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)
    
    # loop through all sessions
    sessions = ['766640955', '767871931', '768515987', '771160300', '771990200', 
                '774875821', '778240327', '778998620', '779839471', '781842082', 
                '786091066', '787025148', '789848216', '793224716', '794812542', 
                '816200189', '821695405', '829720705', '831882777', '835479236', 
                '839068429', '840012044', '847657808']
    
    for i_session, session_id in sessions:

        # display progress
        t_start_s = timer()
        print(f"\nAnalyzing session {i_session+1}/{len(sessions)}")
        print(f"    {session_id}")

        # load session data
        session = cache.get_session_data(session_id)

        # init Neo Block
        block = neo.Block()
        segment = neo.Segment()

        # add spike data to block
        for brain_structure in BRAIN_STRUCTURES:
            spike_trains = create_neo_spiketrains(session, brain_structure)
            segment.spiketrains.append(spike_trains)

        # add running wheel data to block
        running_speed, time_rs = compute_running_speed(session, FS_RUNNING)
        running_speed_as = neo.AnalogSignal(running_speed, units=pq.CompoundUnit("cm/s"), sampling_rate=FS_RUNNING*pq.Hz, name='running_speed')
        segment.analogsignals.append(running_speed_as)

        # add pupil data to block
        pupil_area, time_pa = compute_pupil_area(session, FS_PUPIL)
        pupil_area_as = neo.AnalogSignal(pupil_area, units=pq.cm**2, sampling_rate=FS_PUPIL*pq.Hz, name='pupil_area')
        segment.analogsignals.append(pupil_area_as)

        # save results
        fname_out = f"block_{session}.pkl"
        save_pkl(block, f"{dir_results}/{fname_out}")

        # display progress
        hour, min, sec = hour_min_sec(timer() - t_start_s)
        print(f"    file complete in {hour} hour, {min} min, and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


if __name__ == '__main__':
    main()