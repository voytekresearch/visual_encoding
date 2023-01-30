"""
Compute velocity time-series for an experimetnal block. 
A median filter can be applied to smooth data.

"""


import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding"
RELATIVE_PATH_OUT = "data/behavior/running/natural_movie_one_more_repeats" # where to save output relative to both paths above
REPO_PATH = "C:/Users/User/visual_encoding"

# Import custom functions
import sys
sys.path.append(REPO_PATH)
from allen_vc.utils import save_pkl

# settings - data of interest
STIMULUS_NAME = 'natural_movie_one_more_repeats' # name of stimulus in allen dataset

# settings - dataset details
FS = 1250 # sampling frequency for interpolation

# settings - analysis
SMOOTH = True # whether to smooth data (median filter)
KERNEL_SIZE = 1*FS # kernel size for median filter

#Make sure epoch lengths are in order least to greatest
def main():
    # identify / create directories
    dir_input = PROJECT_PATH + '/data/behavior/running/session_timeseries'
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)

    # load project cache
    manifest_path = f"{MANIFEST_PATH}/manifest_files"
    cache = EcephysProjectCache.from_warehouse(manifest=f"{manifest_path}/manifest.json")
    sessions = cache.get_session_table()

    # Iterate over each session
    for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
        # get session data and display progress 
        print(f'Analyzing session: \t{session_id}')
        session = cache.get_session_data(session_id)

        # load session velocity timeseries
        fname = f'running_{session_id}.npz'
        data_in = np.load(f'{dir_input}/{fname}')

        # get running data for stimulus block
        results = get_stimulus_block_behavioral_series(STIMULUS_NAME, session, data_in['time'], \
            data_in['velocity'], smooth=SMOOTH, kernel_size=KERNEL_SIZE)

        # save results
        fname_out = f"{fname[:-4]}.pkl"
        for base_path in [PROJECT_PATH, MANIFEST_PATH]:
            dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
            save_pkl(results, f"{dir_results}/{fname_out}")


def get_stimulus_block_behavioral_series(stimulus_name, session, time, velocity, \
    smooth=True, kernel_size=None):
    """
    Retrieves the timeseries of the velocity of an animal's behavior during 
    a stimulus presentation.

    Parameters
    ----------
    stimulus_name (str): The name of the stimulus being presented.
    session (Allen session object): session containing the behavioral data.
    time (array-like): The time array corresponding to the velocity data.
    velocity (array-like): The velocity array corresponding to the time data.
    block (int): The index of the stimulus block. Default is 0.
    smooth (boolean): If True, the data will be smoothed with median_filter. Default is True.
    kernel_size (int): The size of the kernel to use for median_filter. Default is None.

    Returns
    -------
    Time, speed, and filtered speed arrays
    """
    # imports
    import neo
    import quantities as pq
    from scipy.ndimage import median_filter

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
        stim_speed = velocity[epoch_mask]

        # Apply a median filter
        if smooth:
        # make sure kernel size is odd
            if kernel_size is None:
                print("Please provide kernel_size")
            else:
                if kernel_size % 2 == 0:
                    ks = kernel_size + 1
            # filter
            stim_speed_filt = median_filter(stim_speed, ks)
        else:
            stim_speed_filt = None

        # return stim_time, stim_speed, stim_speed_filt

        # convert to neo.AnalogSignal
        output = neo.AnalogSignal(stim_speed_filt*(pq.cm/pq.s), sampling_rate=FS*pq.Hz, \
            block=block, t_start=start_time*pq.s)
        stim_group.analogsignals.append(output)

    return stim_group



if __name__ == "__main__":
    main()