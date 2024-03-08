"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

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
from paths import PATH_EXTERNAL
from analysis import compute_pyspike_metrics, compute_cv, compute_fano_factor
from neo_utils import combine_spiketrains

def main():
    # Define/create directories for inputs/outputs
    dir_input = f"{PATH_EXTERNAL}/data/blocks/segmented/{STIM_CODE}"
    files = os.listdir(dir_input)
    
    dir_output = f"{PATH_EXTERNAL}/data/spike_stats"
    for folder in ['region_metrics', 'unit_metrics']:
        if not os.path.exists(f"{dir_output}/{folder}"): 
            os.makedirs(f"{dir_output}/{folder}")

    # initialize data frame
    columns = ['session', 'brain_structure', 'epoch_idx', 'epoch_times', 'running',
               'region_cv', 'spike_distance', 'spike_synchrony', 
               'mean_firing_rate', 'mean_fano_factor', 'mean_cv',
               'firing_rate', 'coef_variation', 'fano_factor', 'unit_index']
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
                    firing_rate = [np.nan] * len(spiketrains)
                    coef_variation = [np.nan] * len(spiketrains)
                    fano_factor = [np.nan] * len(spiketrains)

                # calculate metrics
                else:
                    # calculate region metrics
                    metrics = list(compute_synchrony(spiketrains))

                    # calculate unit metrics
                    firing_rate = [len(spiketrain)/float(spiketrain.duration.item()) for spiketrain in spiketrains]
                    coef_variation = [compute_cv(spiketrain) for spiketrain in spiketrains]
                    fano_factor = [compute_fano_factor(spiketrain, bin_size=1*pq.s) for spiketrain in spiketrains]
                
                # add to data frame
                data = [session, structure, i_seg, [segment.t_start.item(), segment.t_stop.item()],
                        segment.annotations['running']]
                data.extend(metrics)
                data.extend([np.mean(firing_rate), np.mean(fano_factor), np.mean(coef_variation),
                             firing_rate, coef_variation, fano_factor, range(len(spiketrains))])
                df_i = pd.DataFrame([data], columns=columns)
                df = pd.concat([df, df_i], axis=0, ignore_index=True)

    # save region data frame
    df_region = df.drop(columns=['firing_rate', 'coef_variation', 'fano_factor', 'unit_index'])
    df_region.to_csv(f'{dir_output}/region_metrics/{STIM_CODE}.csv', index=False)

    # save unit data frame
    df_units = df.drop(columns=['region_cv', 'spike_distance', 'spike_synchrony',
                                'mean_firing_rate', 'mean_fano_factor', 'mean_cv'])
    df_units = df_units.explode(['firing_rate', 'coef_variation', 'fano_factor', 'unit_index'])
    df_units.to_csv(f'{dir_output}/unit_metrics/{STIM_CODE}.csv', index=False)

def compute_synchrony(spiketrains):
    """
    calculate spike synchrony metrics for a population of spike trains,
    including coefficient of variance, SPIKE-distance, and SPIKE-synchrony.

    Parameters
    ----------
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object

    Returns
    -------
    coeff_of_var: float
        coefficient of variation over all units during specified epoch.
    spike_dist: float
        SPIKE-distance (pyspike) over all units during specified epoch.
    spike_sync: float
        SPIKE-synchrony (pyspike) over all units during specified epoch.
    """

    # combine spiketrains
    region_spiketrain = combine_spiketrains(spiketrains, t_stop=spiketrains[0].t_stop)
    
    # compute coefficient of variation
    coeff_of_var = compute_cv(region_spiketrain)

    # compute spike-synchrony and spike-distance (suppress print statements. bug?)
    sys.stdout = open(os.devnull, 'w')
    spike_sync, spike_dist = compute_pyspike_metrics(spiketrains)
    sys.stdout = sys.__stdout__

    return coeff_of_var, spike_dist, spike_sync


if __name__ == '__main__':
    main()







    