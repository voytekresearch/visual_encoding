"""
Compute velocity time-series for an experimetnal block. 
A median filter can be applied to smooth data.

"""
# Settings - directories
PROJECT_PATH = "G:/Shared drives/visual_encoding" # shared results directory
MANIFEST_PATH = "C:/datasets/allen_vc/manifest_files" # local dataset directory
STIM_CODE = 'spontaneous' # name for output folder (stimulus of interest)

# settings - data of interest
STIMULUS_NAME = 'spontaneous' # name of stimulus in allen dataset

# settings - dataset details
PF = 50 # sampling frequency for interpolation

# settings - analysis
SMOOTH = True # whether to smooth data (median filter)
KERNEL_SIZE = 1*PF # kernel size for median filter

# Imports - general
import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import quantities as pq
from scipy.ndimage import median_filter
from time import time as timer

# Imports - custom
import sys
sys.path.append('allen_vc')
from utils import save_pkl, hour_min_sec
print('Imports complete.')

def main():
    # time it
    t_start = timer()

    # identify / create directories
    dir_input = f"{PROJECT_PATH}/data/behavior/pupil/session_timeseries"
    dir_results = f"{PROJECT_PATH}/data/behavior/pupil/{STIM_CODE}"
    if not os.path.exists(dir_results): 
        os.makedirs(dir_results)

    # load project cache
    cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")
    sessions = cache.get_session_table()
    print('Project cache loaded.')

    # Iterate over each session
    session_ids = sessions[sessions.get('session_type')=='functional_connectivity'].index
    print(f'{len(session_ids)} sessions identified...')
    for i_session, session_id in enumerate(session_ids):
        if not os.path.exists(f'{dir_input}/pupil_area_{session_id}.npz'):
            continue

        # display progress
        t_start_i = timer()
        print(f"    Analyzing session: {session_id} ({i_session+1}/{len(session_ids)})")

        # get session data
        session = cache.get_session_data(session_id)

        # load session velocity timeseries
        fname = f'pupil_area_{session_id}.npz'
        data_in = np.load(f'{dir_input}/{fname}')

        # get pupil data for stimulus block
        results = get_stimulus_block_behavioral_series(STIMULUS_NAME, session, data_in['time'], \
            data_in['velocity'], smooth=SMOOTH, kernel_size=KERNEL_SIZE)

        # save results
        fname_out = fname.replace('.npz','.pkl')
        save_pkl(results, f"{dir_results}/{fname_out}")

        # display progress
        _, min, sec = hour_min_sec(timer() - t_start_i)
        print(f"\tsession complete in {min} min and {sec :0.1f} s")

    # display progress
    hour, min, sec = hour_min_sec(timer() - t_start)
    print(f"\n\n Total Time: \t {hour} hours, {min} minutes, {sec :0.1f} seconds")


def get_stimulus_block_behavioral_series(stimulus_name, session, time, pupil_area, \
    smooth=True, kernel_size=None):
    """
    Retrieves the timeseries of the velocity of an animal's behavior during 
    a stimulus presentation.

    Parameters
    ----------
    stimulus_name : str
        The name of the stimulus being presented.
    session (Allen session object): session containing the behavioral data.
    time : array-like
        The time array corresponding to the velocity data.
    velocity : array-like 
        The velocity array corresponding to the time data.
    block int: 
        The index of the stimulus block. Default is 0.
    smooth : boolean 
        If True, the data will be smoothed with median_filter. Default is True.
    kernel_size : int
        The size of the kernel to use for median_filter. Default is None.

    Returns
    -------
    Time, speed, and filtered speed arrays
    """

    # get start and stop time of stimulus block (0-indexec)
    stimuli_df = session.get_stimulus_epochs()
    stimuli_df = stimuli_df[stimuli_df.get('stimulus_name')==stimulus_name].\
    get(['start_time','stop_time'])

    # retrieve data for each block and append to group
    group_name = f"{STIMULUS_NAME}_filtered"
    stim_group = neo.Group(name=group_name)

    for block in range(len(stimuli_df)):
        start_time = stimuli_df.get('start_time').iloc[block]
        stop_time = stimuli_df.get('stop_time').iloc[block]

        # epoch data
        epoch_mask = (time>start_time) & (time<stop_time)
        stim_time = time[epoch_mask]
        stim_pupil_area = pupil_area[epoch_mask]

        # Apply a median filter
        if smooth:
        # make sure kernel size is odd
            if kernel_size is None:
                print("Please provide kernel_size")
            else:
                if kernel_size % 2 == 0:
                    ks = kernel_size + 1
            # filter
            stim_pupil_filt = median_filter(stim_pupil_area, ks)
        else:
            stim_pupil_filt = None

        # return stim_time, stim_speed, stim_speed_filt

        # convert to neo.AnalogSignal
        output = neo.AnalogSignal(stim_pupil_filt*(pq.cm/pq.s), sampling_rate=PF*pq.Hz, \
            block=block, t_start=start_time*pq.s)
        stim_group.analogsignals.append(output)

    return stim_group


if __name__ == "__main__":
    main()