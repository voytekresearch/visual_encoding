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
    for t_start in np.array(epochs[:,0]):
        ax.axvline(t_start, color='b')
    for t_stop in np.array(epochs[:,1]):
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


def running_segment_plot(block, title):
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
    run_proportions = []
    running = block.annotations['running']
    
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True, figsize=(14,6))
    
    for i_seg in range(len(block.segments)):
        # get running speed for segment
        data = block.segments[i_seg].analogsignals[1]
        speed = data.magnitude
        
        state = block.segments[i_seg].annotations['running']
        
        if state:
            run_proportions.append(sum(np.hstack(speed) > 1)/len(np.hstack(speed)))

        # plot speed
        ax[int(state)].plot(speed)
        
    # print number of running and stationary segments
    print(f"Running segments: {int(np.sum(running))}")
    print(f"Stationary segments: {int(len(running)-np.sum(running))}")
    print(f"Proportions of time running in run segments: \n\n{run_proportions}\n")
    print(f"Average proportion: {np.mean(run_proportions)}\n\n")
    
    plt.title(title)

