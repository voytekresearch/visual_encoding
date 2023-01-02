"""
Ploting utility function
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

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

def sync_plot(df, metrics, condition):
    """
    Plot violin plots for each spike statistic in the given dataframe (df) for the given condition.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to be plotted.
    metrics : list
        List of spike statistics to be plotted.
    condition : str
        Condition to be plotted.

    Returns
    -------
    None
    """
    import seaborn as sns
    # plot violin plots for each spike statistic
    for metric in metrics:
    # set plotting parameters
        plotting_params = {
            'data':    df,
            'x':       'brain_structure',
            'hue':     condition,
            'y':       metric,
            'split':   True
        }

        # create figure
        fig, ax = plt.subplots()
        plt.title(f'{metric}')
        vp = sns.violinplot(**plotting_params, ax=ax, color = 'magenta')
        sp = sns.swarmplot(**plotting_params, ax=ax, color=[0,0,0])