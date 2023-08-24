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

def sync_plot(df, metrics, condition, markersize=5, fname_out=None):
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
    markersize : int, optional
        Size of the markers in the swarm plot. Default is 5.
    fname_out : str, optional
        File path for figure to be saved out to.

    Returns
    -------
    None
    """

    # imports
    import seaborn as sns

    regions = df['brain_structure'].unique()

    # plot violin plots for each spike statistic
    fig, ax = plt.subplots(len(regions), len(metrics), figsize=(12,8), sharex=True, sharey=True)
    plt.xlabel('region')

    for j, region in enumerate(regions):
        # segment data by region
        region_df = df[df['brain_structure'] == region]

        # plot each metric
        for i, metric in enumerate(metrics):
            # set plotting parameters
            plotting_params = {
                'data':    region_df,
                'x':       'block',
                'hue':     condition,
                'y':       metric#,
                #'split':   False
            }

            # add data
            vp = sns.violinplot(**plotting_params, ax=ax[i,j], palette='Blues')
            sp = sns.swarmplot(**plotting_params, dodge=True, ax=ax[i,j], color=[0,0,0], size=markersize)
            ax[i,j].get_legend().remove()
            ax[i,j].set(xlabel=None, ylabel=None)

            # add legend on final plot only
            if i==len(metrics)-1 and j==len(regions)-1:
                handles, _ = vp.get_legend_handles_labels()
                labels = region_df[condition].unique().tolist()
                vp.legend(handles=handles, labels=labels)

            # label figure on edges only
            if j==0:
                ax[i,j].set_ylabel(' '.join(metric.split('_')))
            if i==len(metrics)-1:
                ax[i,j].set_xlabel(region)

    # save figure
    if not fname_out is None:
        fig.savefig(fname_out)


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


def running_segment_plot(block, title=None, fname_out=None, show=True, verbose=False):
    """
    Plot the running speed of a given block of data.

    Parameters
    ----------
    block : ndx.Block
        Block of data to plot running speed.
    title : str
        Title of the plot.
    fname_out : str
        Filename of the output figure.
    show : bool
        Whether to show the figure or not.
    verbose : bool
        Whether to print information about the block.

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
    if verbose:
        print(f"Running segments: {int(np.sum(running))}")
        print(f"Stationary segments: {int(len(running)-np.sum(running))}")
        print(f"Proportions of time running in run segments: \n\n{run_proportions}\n")
        print(f"Average proportion: {np.mean(run_proportions)}\n\n")
    
    if title is not None:
        plt.suptitle(title)

    # save/show figure
    if not fname_out is None:
        fig.savefig(fname_out)
    if show:
        plt.show()
    else:
        plt.close()


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
    st_visp = segment.filter(objects=neo.SpikeTrain, targdict={'brain_structure': 'VISp'})
    st_lgd = segment.filter(objects=neo.SpikeTrain, targdict={'brain_structure': 'LGd'})
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



def plot_linregress(df, x_data, y_data, group=None, multireg=False, title=None, fname_out=None, show=False):
    """
    Calculate and plot the linear regression of two columns in a dataframe.

    Parameters
    ----------
    x_data : str
        column with x-values of dataset
    y_data : str
        column with y-values of dataset
    group: str, optional
        column to color/section data by
    multireg: bool, optional
        whether or not to plot regression lines for each group
    title : str, optional
        Title of the plot
    fname_out : str, optional
        Filename of the output figure
    show : bool, optional
        Whether to show the figure or not

    Returns
    -------
    None
    """
    
    # create figure
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    fig.patch.set_facecolor('white') # set background color to white for text legibility
    
    # plot data
    if group is not None:
        groups = df[group].unique()
        for i, g in enumerate(groups):
            gdf = df[df[group] == g]
            ax.scatter(gdf[x_data], gdf[y_data], label=g, alpha=0.6)

            if multireg:
                # run regression and plot results
                plot_regression_line(gdf[x_data], gdf[y_data], ax=ax, text_height=0.9-i*0.2)

    else:
        ax.scatter(df[x_data], df[y_data])

    if not multireg:
        # run regression and plot results
        plot_regression_line(df[x_data], df[y_data], ax=ax)

    # label figure
    ax.legend()
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


def plot_regression_line(x, y, ax, text_height=0.9):
    """
    Plot the linear regression of two columns in a dataframe on existing axis.

    Parameters
    ----------
    x : array-like
        x-values of dataset
    y : array-like
        y-values of dataset
    ax : matplotlib.axis.Axis
        axis which to plot the data on
    text_height : float, optional
        height of regression report on the right of plot

    Returns
    -------
    None
    """

    # imports
    from scipy.stats import linregress

    results = linregress(x, y)
    t_lin = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    lin = results.slope * t_lin + results.intercept
    ax.plot(t_lin, lin, linewidth=5, color='black')
    ax.plot(t_lin, lin, linewidth=3)

    # add regression results text
    if results.pvalue < 0.001:
        pval = f"{results.pvalue:.2e}"
    else:
        pval = f"{results.pvalue:.3f}"
    plt.text(1.05, text_height, 
             f"Regression \n" +
             f"    Slope: {results.slope:.3f}\n" +
             f"    Intercept: {results.intercept:.3f}\n" +
             f"    R: {results.rvalue:.3f}\n" +
             f"    p: {pval}", transform = ax.transAxes, fontsize=12)



def plot_analog_signal(signal, ax=None, title=None, y_label=None, fname=None):
    """
    Plot a Neo AnalogSignal object.

    Parameters
    ----------
    signal : AnalogSignal
        analog signal to plot
    ax : matplotlib.axes.Axes, optional
        axes to plot on
    title : str, optional
        title of the plot
    y_label : str, optional
        y-axis label
    fname : str, optional
        filename of the output figure. If None, figure will not be saved.

    Returns
    -------
    None
    """

    # init figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=[8,4])

    # plot signal
    ax.plot(signal.times, signal)

    # set title
    if title is not None:
        ax.set_title(title)

    # set axis labels
    if y_label is None:
        if signal.name is not None:
            y_label = signal.name
        else:
            y_label = 'signal'
    ax.set(xlabel=f"time ({signal.times.units.dimensionality.string})", \
        ylabel=f"{y_label} ({signal.units.dimensionality.string})")

    # save
    if fname is not None:
        plt.savefig(fname)


def plot_time_resolved_params(df, window_size, title=None):
    """
    Plot normalized time resolved aperiodic parameters.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing lfp parameter data.
    window_size: float
        window size used in time resolved parameter calculation
    title : str, optional
        title of the produced plot

    Returns
    -------
    None
    """

    sessions = df['session'].unique()

    for session in sessions:
        sdf = df[df['session'] == session]
        t = np.linspace(0, len(sdf)*window_size, len(sdf))

        fig, ax = plt.subplots()
        plt.set_cmap('Blues')
        for i, series in enumerate(['avg_pupil_area', 'inst_spike_rate', 'exponent', 'offset']):
            ax.plot(t, np.array((sdf[series] - sdf[series].mean())/sdf[series].std()) + (3-i)*4, alpha=(1-0.2*i), label=series)

        ax.set(xlabel=f"time (s)", ylabel="normalized parameters (AU)")
        ax.set_title(str(session))
        ax.legend()

        plt.show()