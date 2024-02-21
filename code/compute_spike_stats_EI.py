"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs, filtering by excitatory and inhibitory units.
"""

# SET PATH
PROJECT_PATH = r"D:/visual_encoding"

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
from analysis import compute_pyspike_metrics, compute_cv, compute_fano_factor
from neo_utils import combine_spiketrains

def main():
    # Define/create directories for inputs/outputs
    dir_input = f"{PROJECT_PATH}/data/blocks/segmented/{STIM_CODE}"
    files = os.listdir(dir_input)
    
    dir_output = f"{PROJECT_PATH}/data/spike_stats/by_cell_type" # add output dir
    for folder in ['pop_metrics', 'unit_metrics']:
        if not os.path.exists(f"{dir_output}/{folder}"): 
            os.makedirs(f"{dir_output}/{folder}")

    # load cell class data
    cell_type_df = pd.read_csv('../') # load unit cell type classification

    # initialize data frame
    columns = ['session', 'epoch_idx', 'epoch_times', 'running',
               'region_cv', 'spike_distance', 'spike_synchrony', 
               'mean_firing_rate', 'mean_fano_factor', 'mean_cv', 'cell_type', # include cell type column
               'firing_rate', 'coef_variation', 'fano_factor', 'unit_index']
    df = pd.DataFrame(columns=columns)

    # loop through files
    for i_file, fname in enumerate(files):
        session = fname.split('_')[1].split('.')[0]
        print(f"\nAnalyzing Session: {session} ({i_file+1}/{len(files)})")

        # load block
        block = neo.io.NeoMatlabIO(f"{dir_input}/{fname}").read_block()

        # filter the block based on unit id
        for cell_type in cell_type_df['cell_type'].unique():
            
            units = cell_type_df[cell_type_df['cell_type'] == cell_type]['unit_id']

            # Calculate spike metrics for each segment
            for i_seg, segment in enumerate(block.segments):
                # get brain structures and ensure it is a list
                brain_structures = block.annotations['spike_brain_structures']
                if isinstance(brain_structures, str):
                    brain_structures = [brain_structures]
            
                # filter for spikes in structure
                spiketrains = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': structure})

                # filter/annotate the spikes based on unit_id
                spiketrains.filter(unit_id in units).annotate(cell_type=cell_type)
                cell_type_trains = spiketrains.filter(cell_type=cell_type)

                # ensure there are spikes in structure
                if len(cell_type_trains) == 0:
                    metrics = [np.nan] * (len(columns)-5)
                    firing_rate = [np.nan] * len(cell_type_trains)
                    coef_variation = [np.nan] * len(cell_type_trains)
                    fano_factor = [np.nan] * len(cell_type_trains)

                # calculate metrics
                else:
                    # calculate region metrics
                    metrics = list(compute_synchrony(cell_type_trains))

                    # calculate unit metrics
                    firing_rate = [len(spiketrain)/float(spiketrain.duration.item()) for spiketrain in cell_type_trains]
                    coef_variation = [compute_cv(spiketrain) for spiketrain in spiketrains]
                    fano_factor = [compute_fano_factor(spiketrain, bin_size=1*pq.s) for spiketrain in cell_type_trains]
                
                # add to data frame
                data = [session, i_seg, [segment.t_start.item(), segment.t_stop.item()],
                        segment.annotations['running']]
                data.extend(metrics)
                data.extend([np.mean(firing_rate), np.mean(fano_factor), np.mean(coef_variation), cell_type,
                            firing_rate, coef_variation, fano_factor, range(len(spiketrains))])
                df_i = pd.DataFrame([data], columns=columns)
                df = pd.concat([df, df_i], axis=0, ignore_index=True)

    # save region data frame
    df_region = df.drop(columns=['firing_rate', 'coef_variation', 'fano_factor', 'unit_index'])
    df_region.to_csv(f'{dir_output}/pop_metrics/{STIM_CODE}.csv', index=False)

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







    