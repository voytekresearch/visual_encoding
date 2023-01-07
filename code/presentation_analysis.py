import os
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings (see Behavioral_Series.py and Valid_Epochs.py for details on parameters)
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding"
REPO_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC = f'{PROJECT_PATH}/data/behavior/running/epoch_times'
N_BLOCKS = 2

# import custom functions
import sys
sys.path.append(REPO_PATH)
from allen_vc.epoch_extraction_tools import get_movie_times

#Make sure epoch lengths are in order least to greatest
def main():
    # identify / create directories
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = DATA_LOC
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)

    # load project cache
    manifest_path = f"{MANIFEST_PATH}/manifest_files/manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()

    # Load data and initialize storage
    all_epochs  = [np.load(f'{DATA_LOC}/natural_movie_one_more_repeats_0.npz'), \
    np.load(f'{DATA_LOC}/natural_movie_one_more_repeats_1.npz')]
    movie_times = {}
    movie_run_bools = {}

    for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
        # Skip messed up session for now
        if session_id==793224716:
            continue

        # get session data and display progress 
        print(f'Analyzing session: \t{session_id}')
        session = cache.get_session_data(session_id)

        for block in range(N_BLOCKS): # two blocks of natural movie one exist for each session
            epochs = all_epochs[block]

            # calculate threshold crossings and individual movie times
            mt = get_movie_times(session, block)
            for i in range(30):
                movie_times[f'{session_id}_{(block*30)+i}'] = np.array([mt[i], mt[i+1]])

            # filter diffs for each start/stop movie time and create/save boolean array
            movie_run_bool = []
            epoch = epochs[f'{session_id}_running']

            if len(epoch)==0:
                    movie_run_bool = [False]*30
            else:
                mt[0] = epoch[0][0]
                for t in range(len(mt)-1):
                    if len(epoch[(epoch>mt[t]) & (epoch<mt[t+1])])==0:
                        if epoch[epoch<=mt[t]][-1] in epoch[:,0]:
                            movie_run_bool.append(True)
                        else:
                            movie_run_bool.append(False)
                    else:
                        movie_run_bool.append(None)

            if block==0:
                movie_run_bools[str(session_id)] = np.array(movie_run_bool)
            else:
                movie_run_bools[str(session_id)] = \
                np.append(movie_run_bools[str(session_id)], np.array(movie_run_bool))

    # save running/movie information
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        np.savez(DATA_LOC + f'/natural_movie_one_running_bool.npz', **movie_run_bools)
        np.savez(DATA_LOC + f'/natural_movie_one_times.npz', **movie_times)


if __name__ == "__main__":
    main()