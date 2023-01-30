"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

import os
import numpy as np
import pandas as pd
import pickle
import quantities as pq

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = r"G:\Shared drives\visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding

# settings - data of interest
BRAIN_STRUCTURES = ['VISp', 'LGd']
STIMULUS_NAME = "natural_movie_one_more_repeats"

# Import custom functions
import sys
sys.path.append(REPO_PATH)
from allen_vc.analysis import calculate_spike_metrics

def main():
    #Initialize space for data storage
    
    data_mat = [0]*9

    for structure in BRAIN_STRUCTURES:
        print(f"\nAnalyzing Region:\t{structure}")
        meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

        # Iterate over each session
        for session_id in meta.get('ecephys_session_id').unique():
            print(f"\n\tAnalyzing Session:\t{session_id}")

            # skip over weird session
            if (STIMULUS_NAME == 'natural_movie_one_more_repeats' or \
                STIMULUS_NAME == 'natural_movie_one_more_repeats_all') and \
            session_id==793224716:
                continue

            # not all sessions have been loaded yet so put this in for safety
            if not os.path.exists(f"{PROJECT_PATH}/data/blocks/{session_id}_{STIMULUS_NAME}.pkl"):
                continue

            block = pd.read_pickle(f"{PROJECT_PATH}/data/blocks/{session_id}_{STIMULUS_NAME}.pkl")

            # Calculate spike metrics for both states
            for segment in block.segments:

                spikes = list(segment.filter(region=structure))

                # Calculate/combine metrics for storage
                metrics = list(calculate_spike_metrics(spikes))
                metrics += [list((segment.t_start, segment.t_stop)), segment.name, structure, session_id]
                data_mat = np.vstack((data_mat, metrics))

    #Save DataFrame including all metrics
    data_mat = np.delete(data_mat, (0), axis=0)
    labels = ['mean_firing_rate', 'coefficient_of_variation', \
    'SPIKE-distance','SPIKE-synchrony','correlation coefficient', \
    'epoch', 'state', 'brain_structure','session_id']

    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        fname_out = "-".join(BRAIN_STRUCTURES) + "_" + STIMULUS_NAME
        pd.DataFrame(data=data_mat, columns=labels).to_csv(f'{PROJECT_PATH}/data/spike_data'+\
            f'/synchrony_data/{fname_out}.csv', index=False)

if __name__ == '__main__':
    main()







    