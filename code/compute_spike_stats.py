"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

import os
import numpy as np
import pandas as pd
import neo

# settings - directories
PROJECT_PATH = r"G:\Shared drives\visual_encoding"

# settings - data of interest
BRAIN_STRUCTURES = ['VISp', 'LGd'] # TEMP
STIM_CODE = "natural_movie_one_more_repeats"

# Import custom functions
import sys
sys.path.append('allen_vc')
from analysis import calculate_spike_metrics

def main():
    # Define/create directories for inputs/outputs
    # dir_input = f"{PROJECT_PATH}/data/blocks/segmented/{STIM_CODE}"
    dir_input = f"{PROJECT_PATH}/data/blocks_segmented/{STIM_CODE}"
    files = os.listdir(dir_input)
    
    dir_output = f"{PROJECT_PATH}/data/spike_stats" 
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # initialize data frame
    columns = ['session', 'brain_structure', 'epoch_idx', 'epoch_times', 
               'mean_firing_rate', 'unit_firing_rates', 'coefficient_of_variation', 
               'spike_distance','spike_synchrony','correlation_coefficient']
    df = pd.DataFrame(columns=columns)

    # loop through files
    for i_file, fname in enumerate(files):
        session = fname.split('_')[1].split('.')[0]
        print(f"\nAnalyzing Session: {session} ({i_file+1}/{len(files)})")

        # load block
        block = pd.read_pickle(f"{dir_input}/{fname}")

        # Calculate spike metrics for each segment
        for i_seg, segment in enumerate(block.segments):
            # loop through brain structures in block
            # brain_structures = block.annotations['spike_brain_structures']
            # for structure in brain_structures:
            for structure in BRAIN_STRUCTURES: # TEMP
                # filter for spikes in structure
                spikes = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': structure})

                # ensure there re spikes in structure
                if len(spikes) == 0:
                    metrics = [np.nan] * (len(columns)-4)

                else:
                    # calculate metrics
                    metrics = list(calculate_spike_metrics(spikes))
                
                # add to data frame
                info = [session, structure, i_seg, [segment.t_start, segment.t_stop]]
                info.extend(metrics)
                df = df.append(pd.DataFrame([info], columns=columns), ignore_index=True)

    # save data frame
    df.to_csv(f'{dir_output}/{STIM_CODE}.csv', index=False)

if __name__ == '__main__':
    main()







    