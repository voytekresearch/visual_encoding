"""
Utility functions for Neo data analysis.

"""

# imports
import numpy as np
import neo


def gen_pop_spiketrain(spike_trains, units='s', t_stop=None):
    """Generates a population spiketrain from a list of individual spike trains.

    Parameters
    ----------
    spike_trains : list
        A list of Neo SpikeTrains 
    units : str, optional
        The units of the spike times (default is 's')
    t_stop : float, optional
        The stop time of the population spike train. If not provided, it will be
        set to the last spike time (default is None)

    Returns
    -------
    pop_spiketrain : neo.SpikeTrain
        A Neo SpikeTrain object with the population spike train
    """
    
    # concatenate spike trains across population
    pop_spikes = np.sort(np.array(np.concatenate(np.array(spike_trains, dtype=object))))

    # get stop time
    if t_stop is None:
        t_stop = pop_spikes[-1]

    # create Neo SpikeTrain for population
    pop_spiketrain = neo.SpikeTrain(pop_spikes, units=units, t_stop=t_stop)
    
    return pop_spiketrain


def get_neo_group_names(block):
    """
    Get the names of all groups in a Neo block.
    
    Parameters
    ----------
    block : neo.Block
        Neo block containing groups.

    Returns
    -------
    group_names : list
        List of group names.   
    """

    group_names = []
    for group in block.groups:
        group_names.append(group.name)
    
    return group_names