"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

# SET PATH
PROJECT_PATH = r"G:\Shared drives\visual_encoding"

# settings - data of interest
STIM_CODE = "spontaneous_stationary"

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
    for folder in ['region_metrics', 'unit_rates']:
        if not os.path.exists(f"{dir_output}/{folder}"): 
            os.makedirs(f"{dir_output}/{folder}")

    # initialize data frame
    columns = ['session', 'brain_structure', 'epoch_idx', 'epoch_times', 'running',
               'mean_firing_rate', 'coefficient_of_variation', 
               'spike_distance','spike_synchrony', 'firing_rate', 'unit_index']
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
                spiketrains = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': structure})

                # ensure there are spikes in structure
                if len(spiketrains) == 0:
                    metrics = [np.nan] * (len(columns)-5)

                # calculate metrics
                else:
                    metrics = list(calculate_spike_metrics(spiketrains))
                    firing_rate = [len(spiketrain)/float(spiketrain.duration.item()) for spiketrain in spiketrains]
                    unit_index = range(len(spiketrains))
                
                # add to data frame
                info = [session, structure, i_seg, [segment.t_start.item(), segment.t_stop.item()],
                        segment.annotations['running']]
                info.extend(metrics)
                info.extend([firing_rate, unit_index])
                df = df.append(pd.DataFrame([info], columns=columns), ignore_index=True)

    # save region data frame
    df_region = df.drop(columns=['firing_rate', 'unit_index'])
    df_region.to_csv(f'{dir_output}/region_metrics/{STIM_CODE}.csv', index=False)

    # save unit data frame
    df_units = df[['session', 'brain_structure', 'epoch_idx', 'unit_index', 'epoch_times', 
                   'running', 'firing_rate']]
    df_units = df_units.explode(['firing_rate', 'unit_index']).reset_index(drop=True)
    df_units.to_csv(f'{dir_output}/unit_rates/{STIM_CODE}.csv', index=False)

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
    
    # compute mean firing rate and coefficient of variation
    mean_firing_rate = len(region_spiketrain) / region_spiketrain.duration.item() / len(spiketrains)
    coeff_of_var = compute_cv(region_spiketrain)

    # compute spike-synchrony and spike-distance (suppress print statements. bug?)
    sys.stdout = open(os.devnull, 'w')
    spike_sync, spike_dist = compute_pyspike_metrics(spiketrains)
    sys.stdout = sys.__stdout__

    return mean_firing_rate, coeff_of_var, spike_dist, spike_sync


if __name__ == '__main__':
    main()







    