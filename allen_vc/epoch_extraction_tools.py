# imports
import numpy as np
import pandas as pd

def get_stimulus_behavioral_series(stimulus_name, session, time, velocity, \
    trial=0, smooth=True, kernel_size=None):
    """
    """

    # get start and stop time of stimulus for correct trial (0-indexec)
    stimuli_df = session.get_stimulus_epochs()
    stimuli_df = stimuli_df[stimuli_df.get('stimulus_name')==stimulus_name].\
    get(['start_time','stop_time'])
    start_time = stimuli_df.get('start_time').iloc[trial]
    stop_time = stimuli_df.get('stop_time').iloc[trial]

    # epoch data
    epoch_mask = (time>start_time) & (time<stop_time)
    spont_time = time[epoch_mask]
    spont_speed = velocity[epoch_mask]

    # Apply a median filter
    if smooth:
    # make sure kernel size is odd
        if kernel_size is None:
            print("Please provide kernel_size")
        else:
            if kernel_size % 2 == 0:
                ks = kernel_size + 1
        # filter
        spont_speed_filt = median_filter(spont_speed, ks)
    else:
        spont_speed_filt = None

    return spont_time, spont_speed, spont_speed_filt


def find_segments(signal, threshold, return_below=False):
    """
    Find segments of a signal that are above/below a threshold.
    
    Parameters
    ----------
        signal : array-like
            The signal to search for segments.
        threshold : float
            Threshold value to search for segments.
        return_below : bool, optional
            If True, return segments below threshold. Default is False.
            
    Returns
    -------
        epoch_times : array-like
            Start and end times of segments.
    """

    # get indices of segments above threshold
    above_threshold = np.where(signal > threshold)[0]
    if len(above_threshold) == 0:
        return np.array([])

    # get start and end of segments
    starts = above_threshold[np.where(np.diff(above_threshold) != 1)[0] + 1]
    ends = above_threshold[np.where(np.diff(above_threshold) != 1)[0]]

    # add first and last index if needed
    if starts[0] > ends[0]:
        starts = np.insert(starts, 0, 0)
    if ends[-1] < starts[-1]:
        ends = np.append(ends, len(signal)-1)

    # join epoch times as array
    epoch_times = np.array([starts, ends]).T

    # print number of epochs dropped
    print(f'Identified {epoch_times.shape[0]} epochs')

    # return segments below threshold if requested
    if return_below:
        epochs_below = np.vstack([np.insert(epoch_times[:,1], 0, 0),
            np.insert(epoch_times[:,0], -1, -1)]).T

        return epoch_times, epochs_below

    else:
        return epoch_times


def plot_epochs(signal, time, epochs, threshold=None):
    """Plots a signal over time, with annotations for epochs.

    Parameters
    ----------
    signal : numpy array
        Signal to be plotted.
    time : numpy array
        Time stamps for the signal.
    epochs : 2D numpy array
        Epochs to annotate.
    threshold : float, optional
        Horizontal line at given value.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Figure and axes for the plot.
    """

    # plot signal
    fig, ax = plt.subplots(figsize=[20,4])
    ax.plot(time, signal)

    # annotate threshold
    if threshold is not None:
        ax.axhline(threshold, color='k')

    # annotate epochs
    for t_start in np.array(time[epochs[:,0]]):
        ax.axvline(t_start, color='b')
    for t_stop in np.array(time[epochs[:,1]]):
        ax.axvline(t_stop, color='r')

    return fig, ax


def join_epochs_with_gap(epochs, min_gap):
    """
    Joins together epochs that have a gap shorter than a given minimum duration between them.

    Parameters
    ----------
    epochs : numpy array
        Nx2 array containing start and end times of each epoch. 
    min_gap : float
        Minimum duration of gap between epochs.

    Returns
    -------
    epochs_clean : numpy array
        Nx2 array containing the start and end times of the remaining epochs.
    """

    epochs_clean = []
    for ii in range(epochs.shape[0] - 1):
        gap = epochs[ii+1, 0] - epochs[ii, 1]

        # if gap is less than the minimun duration
        if gap < min_gap:
            # treat first differenctly
            if epochs_clean==[]:
                epochs_clean.append([epochs[ii, 0], epochs[ii+1, 1]])
            else:
                # check previous entry
                if epochs_clean[-1][1] == epochs[ii, 1]:
                    epochs_clean[-1][1] = epochs[ii+1, 1]
                else:
                    epochs_clean.append([epochs[ii, 0], epochs[ii+1, 1]])

        # if gap is long enough
        else:
            # treat first differenctly
            if epochs_clean==[]:
                epochs_clean.append(epochs[ii])
            else:
                # check previous entry
                if epochs_clean[-1][1] == epochs[ii, 1]:
                    continue
                else:
                    epochs_clean.append(epochs[ii])

    epochs_clean = np.array(epochs_clean)

    # print number of epochs dropped
    print(f'Joined {epochs.shape[0] - epochs_clean.shape[0]} / {epochs.shape[0]} epochs')

    return epochs_clean


def drop_short_epochs(epochs, min_duration):
    """
    Drop epochs shorter than a given duration

    Parameters
    ----------
    epochs : ndarray
        2D array of epochs, of shape (n_epochs, 2).
    min_duration : float
        Minimum duration of epochs to keep.

    Returns
    -------
    epochs_clean : ndarray
        2D array of epochs, with epochs shorter than `min_duration` removed.
    """

    # get duration of epochs
    duration = np.ravel(np.diff(epochs, axis=1))

    # drop epochs below threshold
    epochs_clean = epochs[duration > min_duration]

    # print number of epochs dropped
    print(f'Dropped {epochs.shape[0] - epochs_clean.shape[0]} / {epochs.shape[0]} epochs')

    return epochs_clean


def get_inverse_epochs(epochs, signal):
    """
    Get inverse epochs from a given epoch array and signal.

    Parameters
    ----------
    epochs : array_like
        2-dimensional array of start and stop times of regular epochs.
    signal : array_like
        Signal array representing the signal being analyzed.

    Returns
    -------
    epochs_inv : array_like
        2-dimensional array of start and stop times of the inverse epochs.
    """

    # swap start and stop times
    # start_times = epochs[:,1]
    # stop_times = epochs[:,0]
    start_times = np.insert(epochs[:,1], 0, 0)
    stop_times = np.append(epochs[:,0], len(signal)-1)

    # if first epoch start on first time point, drop first stop time
    if epochs[0,0] == 0:
        start_times = start_times[1:]
        stop_times = stop_times[1:]

    # if last epoch ends on last time point
    if epochs[-1,1] == len(signal)-1:
        start_times = start_times[:-1]
        stop_times = stop_times[:-1]

    # combine epoch times
    epochs_inv = np.vstack([start_times, stop_times]).T

    return epochs_inv


def get_epoch_times(signal, threshold, min_duration):

    # id epochs above threshold
    epochs_above = find_segments(signal, threshold=threshold, return_below=False)

    # join epochs
    epochs_above = join_epochs_with_gap(epochs_above, min_gap=min_duration)

    # drop short epochs
    epochs_above = drop_short_epochs(epochs_above, min_duration=min_duration)
    
    # if no above-threshold epochs identified
    if len(epochs_above) == 0:
        epochs_below = np.array([[0, len(signal)-1]])

    else:
        # get below-threshold epoch times
        epochs_below = get_inverse_epochs(epochs_above, signal)

        # drop short epochs
        epochs_below = drop_short_epochs(epochs_below, min_duration)

    return epochs_above, epochs_below