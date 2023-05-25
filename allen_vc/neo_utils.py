"""
Utility functions for Neo data analysis.

"""

# imports
import numpy as np
import neo


def combine_spiketrains(spiketrain_list, units='s', t_stop=None):
    """Generates a single spiketrain from a list of spiketrains.

    Parameters
    ----------
    spiketrain_list : list
        A list of Neo SpikeTrains 
    units : str, optional
        The units of the spike times (default is 's')
    t_stop : float, optional
        The stop time of the population spike train. If not provided, it will be
        set to the last spike time (default is None)

    Returns
    -------
    spiketrain : neo.SpikeTrain
        A Neo SpikeTrain object with the population spike train
    """
    
    # concatenate spike trains across population
    all_spikes = np.sort(np.array(np.concatenate(spiketrain_list)))

    # get stop time in needed
    if t_stop is None:
        t_stop = all_spikes[-1]

    # create Neo SpikeTrain for population
    spiketrain = neo.SpikeTrain(all_spikes, units=units, t_stop=t_stop)
    
    return spiketrain


def get_group_names(block):
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


def get_analogsignal(block, name, segment_idx=None, return_numpy=True, return_annotations=False):
    """
    Returns an analog signal from a Neo Block object. If multiple segments are
    present, the signal from each segment is returned as a list.

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
        analog signal is returned as a Neo AnalogSignal object or a list of 
        AnalogSignal objects. Default is True.
    return_annotations : bool, optional
        If True, the annotations for the analog signal are returned. Default is False.

    Returns
    -------
    a_signal : array_like or neo.core.AnalogSignal or list
        Analog signal from the Neo Block object. If return_numpy is True,
        the analog signal is returned as a numpy array. If return_numpy is False,
        the analog signal is returned as a Neo AnalogSignal object or a list of
        AnalogSignal objects.

    """
    # get signal index
    signal_names = get_analogsignal_names(block, segment_idx)
    signal_idx = np.argwhere(signal_names == name)[0][0]

    # get analog signal from block for all segments
    if segment_idx is None:
        a_signal = []
        annotations = block.segments[0].analogsignals[signal_idx].annotations
        for segment in block.segments:
            a_signal.append(segment.analogsignals[signal_idx])

        # convert to numpy array
        if return_numpy:
            a_signal_list = []
            for signal in a_signal:
                a_signal_list.append(np.array(signal))
            # join as matrix
            if len(a_signal[0].shape) == 1:
                a_signal = np.concatenate(a_signal_list)
            elif len(a_signal[0].shape) == 2:
                a_signal = np.dstack(a_signal_list)
                a_signal = np.moveaxis(a_signal, 2, 0)
            else:
                raise ValueError('Analog signal has too many dimensions.')

    # get analog signal from block for a single segment
    else:
        a_signal = block.segments[segment_idx].analogsignals[signal_idx]
        annotations = a_signal.annotations

        # convert to numpy array
        if return_numpy:
            a_signal = np.array(a_signal)

    # return
    if return_annotations:
        return a_signal, annotations
    else:
        return a_signal


def get_spike_times(segment, region=None):
    """
    Extract spike times from a Neo Block object. Useful for plotting spike rasters.

    Parameters
    ----------
    block : Neo Segment object
        Neo segment containing spike trains.
    region : str, optional
        Region of interest. Default is None.

    Returns
    -------
    spike_times : list
        List of arrays of spike times for each unit in `region`.
    """

    # get spike times (for region)
    if region is None:
        st = segment.filter(objects=neo.SpikeTrain)
    else:
        st = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': region})

    # convert to list of arrays
    spike_times = [st.times for st in st]

    return spike_times

