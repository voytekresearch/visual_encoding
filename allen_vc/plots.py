"""
Ploting utility function
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# custom imports
import sys
sys.path.append('../')
from allen_vc.neo_utils import get_analogsignal

# Matplotlib rcParams ---------------------------------------------------------
# misc
rcParams['figure.constrained_layout.use'] = True

# font
rcParams['figure.titlesize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10

# Figure output
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'png'

# background color
rcParams['figure.facecolor'] = 'w'
rcParams['axes.facecolor'] = 'w'

# -----------------------------------------------------------------------------

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
    for t_start in np.array(epochs[:,0]):
        ax.axvline(t_start, color='b')
    for t_stop in np.array(epochs[:,1]):
        ax.axvline(t_stop, color='r')

    return fig, ax

def sync_plot(df, metrics, condition, markersize=5):
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
        fig, ax = plt.subplots(figsize=(15,10))
        #plt.title(f'{metric}')
        vp = sns.violinplot(**plotting_params, ax=ax, palette='Blues')
        sp = sns.swarmplot(**plotting_params, ax=ax, color=[0,0,0], size=markersize)
        plt.ylabel(' '.join(metric.split('_')))
        plt.xlabel('brain structure')
        plt.legend(fontsize=15)
        #sp.get_legend().remove()
        # having trouble removing swarmplot legend ONLY


def scatter_2_conditions(x1, y1, x2, y2, conditions=['cond 1', 'cond 2'],
                         title=None, xlabel=None, ylabel=None, fname_out=None, show=False):
    """
    Calculate and plot the linear regression for two datasets, with optional labels and file output.

    Parameters
    ----------
    x1 : 1-d array_like
        x-values of the first dataset
    y1 : 1-d array_like
        y-values of the first dataset
    x2 : 1-d array_like
        x-values of the second dataset
    y2 : 1-d array_like
        y-values of the second dataset
    conditions : list, optional
        List of strings containing the names of the conditions to be plotted.
    title : str, optional
        Title of the plot
    xlabel : str, optional
        x-axis label of the plot
    ylabel : str, optional
        y-axis label of the plot
    fname_out : str, optional
        Filename of the output figure
    show : bool, optional
        Whether to show the figure or not

    Returns
    -------
    None

    Notes
    -----
    The linear slopes, r-values, p-values, and intercepts of both 
    datasets will be printed on the plot.
    """

    # imports
    from scipy.stats import linregress
    
    # create figure
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    fig.patch.set_facecolor('white') # set background color to white for text legibility
    
    # loop through conditions
    for x_data, y_data, label, offset in zip([x1, x2], [y1, y2], conditions, [0, 0.2]):
        # plot data
        ax.scatter(x_data, y_data)

        # run regression and plot results
        results = linregress(x_data, y_data)
        t_lin = np.linspace(np.nanmin(x_data), np.nanmax(x_data), 100)
        lin = results.slope * t_lin + results.intercept
        ax.plot(t_lin, lin, label=label)

        # add regression results text
        if results.pvalue < 0.001:
            pval = f"{results.pvalue:.2e}"
        else:
            pval = f"{results.pvalue:.3f}"
        plt.text(1.05, 0.9 - offset, 
                 f"Regression ({label}):\n" +
                 f"    Slope: {results.slope:.3f}\n" +
                 f"    Intercept: {results.intercept:.3f}\n" +
                 f"    R: {results.rvalue:.3f}\n" +
                 f"    p: {pval}", transform = ax.transAxes)

    # label figure
    ax.legend()
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    # save/show figure
    if not fname_out is None:
        fig.savefig(fname_out)
    if show:
        plt.show()
    else:
        plt.close()

def plot_sa_heat_map(r_mat, xlabels, ylabels, graph_title=None,
                     sig_mask=None, fname_out=None, show_fig=True, vmin=-1, vmax=1):
    """
    Plot a sensitivity analysis heat map with -1 to 1 color map limits.

    Parameters
    ----------
    r_mat : array_like
        2D array of sensitivity analysis results.
    sig_mat : array_like
        2D array of bool values indicating whether a result is significant.
    xlabels: array_like
        labels for x axis
    ylabels: array_like
        labels for y axis
    graph_title (optional): str
        title of heat map created
    sig_mask (optional): array_like
        2D array of bool values indicating whether a result is significant.
        Default is None. If None, all values are plotted.
    fname_out (optional): str
        path to save figure. Default is None. If None, figure is not saved.
    show_fig (optional): bool
        whether to show figure. Default is True.

    Returns
    -------
    None

    """
    import seaborn as sns

    # set non-significant values to 0
    if not sig_mask is None:
        data_plt = np.zeros_like(r_mat)
        data_plt[sig_mask] = r_mat[sig_mask]
    else:
        data_plt = r_mat

    # plot
    plt.figure(figsize=(14,7))
    sns.heatmap(data_plt, xticklabels=xlabels, yticklabels=ylabels, 
                vmin=vmin, vmax=vmax, center=0, cbar_kws={'label': 'Pearson R'}, cmap='coolwarm')
    
    # label
    plt.xlabel("Lower Limit (Hz)")
    plt.ylabel("Upper Limit (Hz)")
    plt.xticks(rotation=30)
    if graph_title:
        plt.title(graph_title)

    # save
    if not fname_out is None:
        plt.savefig(fname_out)

    if show_fig:
        plt.show()
    else:
        plt.close()


def running_segment_plot(block, title=None):
    """
    Plot the running speed of a given block of data.

    Parameters
    ----------
    block : ndx.Block
        Block of data to plot running speed.
    title : str
        Title of the plot.

    Returns
    -------
    None
        Prints the number of running and stationary segments, proportions 
        of time running in run segments, and the average proportion.
    """

    # check for running annotations
    if not 'running' in block.annotations:
        print("No running annotations found.")
        return

    # init
    run_proportions = []
    running = block.annotations['running']

    # create figure
    fig, axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(14,6))
    axes[0].set_title("stationary epochs")
    axes[1].set_title("running epochs")
    
    # loop through segments
    for i_seg in range(len(block.segments)):
        # get running speed for segment
        speed, time  = get_analogsignal(block, 'running_speed', segment_idx=i_seg, 
                                        return_numpy=True, reset_time=True)
        
        # compute proportion of time running
        state = block.segments[i_seg].annotations['running']
        if state:
            run_proportions.append(sum(np.hstack(speed) > 1)/len(np.hstack(speed)))

        # plot speed
        axes[int(state)].plot(time, speed)

    # label plot
    for ax in axes:
        ax.set(xlabel='time (s)', ylabel='speed (cm/s)')
        
    # print number of running and stationary segments
    print(f"Running segments: {int(np.sum(running))}")
    print(f"Stationary segments: {int(len(running)-np.sum(running))}")
    print(f"Proportions of time running in run segments: \n\n{run_proportions}\n")
    print(f"Average proportion: {np.mean(run_proportions)}\n\n")
    
    if title is not None:
        plt.suptitle(title)


def plot_segment(block, i_seg):
    """
    Plot the LFP, spike trains, running speed, and pupil area for a given segment.

    Parameters
    ----------
    block : Neo Block object
        Neo block containing segment to plot.
    i_seg : int
        Index of segment to plot.

    Returns
    -------
    None
    """

    # imports
    from matplotlib import gridspec
    import neo

    # settings
    col_0 = np.array([52,148,124]) /255
    col_1 = np.array([244,157,70]) /255

    # get data of interest
    running_speed, t_running_speed  = get_analogsignal(block, 'running_speed', segment_idx=i_seg, return_numpy=True)
    pupil_area, t_pupil_area = get_analogsignal(block, 'pupil_area', segment_idx=i_seg, return_numpy=True)

    # get spike times for each region
    segment = block.segments[i_seg]
    st_visp = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': 'VISp'})
    st_lgd = segment.filter(objects=neo.SpikeTrain,targdict={'brain_structure': 'LGd'})
    st_visp = [st.times for st in st_visp]
    st_lgd = [st.times for st in st_lgd]

    # create figure and gridspec
    fig = plt.figure(figsize=[8,4])#, constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=5, height_ratios=[1,1,1,1,1])
    ax_a = fig.add_subplot(spec[0,0])
    ax_b = fig.add_subplot(spec[1,0], sharex=ax_a)
    ax_c = fig.add_subplot(spec[2,0], sharex=ax_a)
    ax_d = fig.add_subplot(spec[3,0], sharex=ax_a)
    ax_e = fig.add_subplot(spec[4,0], sharex=ax_a)
    # plot subplot a: LFP

    try:
        lfp, t_lfp = get_analogsignal(block, 'lfp', segment_idx=i_seg, return_numpy=True)
        ax_a.pcolormesh(t_lfp, np.arange(0, lfp.shape[1]), lfp.T, shading='auto')
        ax_a.set_ylabel("LFP", rotation=0, labelpad=40)
    except:
        print("No LFP data found.")

    # plot subplot b: spikes (VISp)
    ax_b.eventplot(st_visp, color='k')
    ax_b.set_ylabel("VISp units", rotation=0, labelpad=40)

    # plot subplot c: spikes (LGd)
    ax_c.eventplot(st_lgd, color='grey')
    ax_c.set_ylabel("LGd units", rotation=0, labelpad=40)

    # plot subplot d: running speed
    ax_d.plot(t_running_speed, running_speed, color=col_0)
    ax_d.set_ylabel("velocity", rotation=0, labelpad=40)

    # plot subplot e : pupil area
    ax_e.plot(t_pupil_area, pupil_area, color=col_1)
    ax_e.set_ylabel("pupil size", rotation=0, labelpad=40)

    # remove axes, axes ticks, and adjust spacing
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
            ax.set_xlabel("") 
            ax.set_yticks([])
            for loc in ['left', 'right', 'top', 'bottom']:
                ax.spines[loc].set_visible(False)
    fig.subplots_adjust(hspace=0)



def plot_linregress(df, x_data, y_data, title=None, fname_out=None, show=False):
    """
    Calculate and plot the linear regression of two columns in a dataframe.

    Parameters
    ----------
    x_data : str
        column with x-values of dataset
    y_data : str
        column with y-values of dataset
    title : str, optional
        Title of the plot
    fname_out : str, optional
        Filename of the output figure
    show : bool, optional
        Whether to show the figure or not

    Returns
    -------
    None

    Notes
    -----
    The linear slopes, r-values, p-values, and intercepts of both 
    datasets will be printed on the plot.
    """

    # imports
    from scipy.stats import linregress
    
    # create figure
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    fig.patch.set_facecolor('white') # set background color to white for text legibility
    
    # plot data
    ax.scatter(df[x_data], df[y_data])

    # run regression and plot results
    results = linregress(df[x_data], df[y_data])
    t_lin = np.linspace(np.nanmin(df[x_data]), np.nanmax(df[x_data]), 100)
    lin = results.slope * t_lin + results.intercept
    ax.plot(t_lin, lin, color='red')

    # add regression results text
    if results.pvalue < 0.001:
        pval = f"{results.pvalue:.2e}"
    else:
        pval = f"{results.pvalue:.3f}"
    plt.text(1.05, 0.9, 
             f"Regression \n" +
             f"    Slope: {results.slope:.3f}\n" +
             f"    Intercept: {results.intercept:.3f}\n" +
             f"    R: {results.rvalue:.3f}\n" +
             f"    p: {pval}", transform = ax.transAxes)

    # label figure
    # ax.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel(x_data)
    plt.ylabel(y_data)
        
    # save/show figure
    if not fname_out is None:
        fig.savefig(fname_out)
    if show:
        plt.show()
    else:
        plt.close()


