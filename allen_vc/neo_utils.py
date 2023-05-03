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


def get_analogsignal_names(block, segment_idx=None, lfp_only=False):
    """
    Returns the names of all analog signals in a given segment Neo Block object. If a 
    segment index is not provided, the names for the first segments are returned.

    Parameters
    ----------
    block : neo.core.Block
        Neo Block object containing the analog signal.
    segment_idx : int, optional
        Index of the segment containing the analog signal. If None, the signal
        names for the first segment are returned. Default is None.
    lfp_only : bool, optional
        If True, only LFP signals are returned. Default is False.

    Returns
    -------
    signal_names : array_like
        Names of the analog signals.
    """

    # get all analog signal names
    if segment_idx is None:
        signal_names = np.array([a_signal.name for a_signal in block.segments[0].analogsignals])
    else:
        signal_names = np.array([a_signal.name for a_signal in block.segments[segment_idx].analogsignals])
    
    # filter for LFP signals
    if lfp_only:
        signal_names = signal_names[np.array(['lfp' in name for name in signal_names])]

    return signal_names


def get_analogsignal(block, name, segment_idx=None, return_numpy=True):
    """
    Returns an analog signal from a Neo Block object. If multiple segments are
    present, the signal from each segment is returned as a list. Output can be
    returned as a numpy array or a Neo AnalogSignal object.

    Parameters
    ----------
    block : neo.core.Block
        Neo Block object containing the analog signal.
    name : str
        Name of the analog signal.
    segment_idx : int, optional
        Index of the segment containing the analog signal. If None, the signal
        from all segments is returned. Default is None.
    return_numpy : bool, optional
        If True, the analog signal is returned as a numpy array. If False, the
        analog signal is returned as a Neo AnalogSignal object. Default is True.

    Returns
    -------
    a_signal : array_like or neo.core.AnalogSignal
        Analog signal from the Neo Block object. The data type depends on the
        value of `return_numpy`.

    """
    # get signal index
    signal_names = get_analogsignal_names(block, segment_idx)
    signal_idx = np.argwhere(signal_names == name)[0][0]

    # get analog signal from block for all segments
    if segment_idx is None:
        a_signal = []
        for segment in block.segments:
            a_sig = segment.analogsignals[signal_idx]

            # convert to numpy array
            if return_numpy:
                a_signal.append(np.array(a_sig).squeeze())
            else:
                a_signal.append(a_sig)
        # join list as 3d array
        a_signal = np.concatenate(a_signal).T


    # get analog signal from block for a single segment
    else:
        a_signal = block.segments[segment_idx].analogsignals[signal_idx]

        # convert to numpy array
        if return_numpy:
            a_signal = np.array(a_signal).squeeze()
        
    return a_signal

