"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

# SET PATH
PROJECT_PATH = r"G:\Shared drives\visual_encoding"

# settings - data of interest
STIM_CODE = "spontaneous_running"

# imports - general
import os
import numpy as np
import pandas as pd
import neo
import quantities as pq

# Import custom functions
import sys
sys.path.append('allen_vc')
from analysis import compute_pyspike_metrics, compute_cv
from neo_utils import combine_spiketrains


def main():
    # Define/create directories for inputs/outputs
    dir_input = f"{PROJECT_PATH}/data/blocks/segmented/{STIM_CODE}"
    files = os.listdir(dir_input)
    
    dir_output = f"{PROJECT_PATH}/data/spike_stats" 
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # initialize data frame
    columns = ['session', 'brain_structure', 'epoch_idx', 'epoch_times', 'running',
               'mean_firing_rate', 'coefficient_of_variation', 
               'spike_distance','spike_synchrony']
    df = pd.DataFrame(columns=columns)

    # loop through files
    for i_file, fname in enumerate(files):
        session = fname.split('_')[1].split('.')[0]
        print(f"\nAnalyzing Session: {session} ({i_file+1}/{len(files)})")

        # load block
        block = neo.io.NeoMatlabIO(f"{dir_input}/{fname}").read_block()

        # Calculate spike metrics for each segment
        for i_seg, segment in enumerate(block.segments):
            # get brain structures and ensure it is a list
            brain_structures = block.annotations['spike_brain_structures']
            if isinstance(brain_structures, str):
                brain_structures = [brain_structures]

            # FIX: remove whitespace from brain structure names
            brain_structures = [structure.strip() for structure in brain_structures]

            # loop through brain structures
            for structure in brain_structures:
                # filter for spikes in structure
                spikes = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': structure})

                # ensure there re spikes in structure
                if len(spikes) == 0:
                    metrics = [np.nan] * (len(columns)-5)

                else:
                    # calculate metrics
                    metrics = list(calculate_spike_metrics(spikes))
                
                # add to data frame
                info = [session, structure, i_seg, [segment.t_start, segment.t_stop], 
                        segment.annotations['running']]
                info.extend(metrics)
                df = df.append(pd.DataFrame([info], columns=columns), ignore_index=True)

    # save data frame
    df.to_csv(f'{dir_output}/{STIM_CODE}.csv', index=False)


def calculate_spike_metrics(spiketrains):
    """
    calculate spike metrics (mean firing rate, coefficient of variance, 
    SPIKE-distance, and SPIKE-synchrony).

    Parameters
    ----------
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object

    Returns
    -------
    mean_firing_rate: float
        mean firing rate over all units during specified epoch.
    coeff_of_var: float
        coefficient of variation over all units during specified epoch.
    spike_dist: float
        SPIKE-distance (pyspike) over all units during specified epoch.
    spike_sync: float
        SPIKE-synchrony (pyspike) over all units during specified epoch.
    """

    # combine spiketrains
    region_spiketrain = combine_spiketrains(spiketrains, t_stop=spiketrains[0].t_stop)
    
    # compute mean firing rate
    mean_firing_rate = len(region_spiketrain) / region_spiketrain.duration / len(spiketrains)

    # compute synchrony metrics
    coeff_of_var = compute_cv(region_spiketrain)
    spike_sync, spike_dist = compute_pyspike_metrics(spiketrains)

    return mean_firing_rate, coeff_of_var, spike_dist, spike_sync


if __name__ == '__main__':
    main()







    