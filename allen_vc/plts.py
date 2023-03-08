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

# compare linear regression plots between movie and shuffled conditions for two variables
def linregress_movie_v_shuffled_plot(x1, y1, x2=None, y2=None, title=None, 
                                     xlabel=None, ylabel=None, fname_out=None, show=False):
    import scipy.stats as sts
    
    #natural_movie
    m1 = sts.linregress(x1, y1)
    l1 = np.linspace(min(x1), max(x1), 1000)
    t1 = m1.slope * l1 + m1.intercept
    
    #shuffled
    m2 = sts.linregress(x2, y2)
    l2 = np.linspace(min(x2), max(x2), 1000)
    t2 = m2.slope * l2 + m2.intercept
    
    plt.style.use('bmh')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    
    plt.scatter(x1, y1)
    plt.plot(l1, t1, label='natural_movie (1)')
    plt.scatter(x2, y2)
    plt.plot(l2, t2, label='shuffled (2)')
    plt.legend()
    
    # add labels
    plt.text(1.05, 0.95, f"Slope1: {round(m1.slope, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.9, f"R1: {round(m1.rvalue, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.85, f"p: {round(m1.pvalue, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.8, f"Intercept1: {round(m1.intercept, 5)}", transform = ax.transAxes)
    
    plt.text(1.05, 0.7, f"Slope2: {round(m2.slope, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.65, f"R2: {round(m2.rvalue, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.6, f"p: {round(m2.pvalue, 5)}", transform = ax.transAxes)
    plt.text(1.05, 0.55, f"Intercept2: {round(m2.intercept, 5)}", transform = ax.transAxes)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
        
    if not fname_out is None:
        plt.savefig(fname_out)
        
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

