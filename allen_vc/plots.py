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

# # misc
# rcParams['figure.constrained_layout.use'] = True

# # font
# rcParams['figure.titlesize'] = 20
# rcParams['axes.titlesize'] = 20
# rcParams['axes.labelsize'] = 14
# rcParams['axes.labelsize'] = 12
# rcParams['xtick.labelsize'] = 10
# rcParams['ytick.labelsize'] = 10
# rcParams['legend.fontsize'] = 10

# # Figure output
# rcParams['savefig.dpi'] = 300
# rcParams['savefig.format'] = 'png'

# # background color
# rcParams['figure.facecolor'] = 'w'
# rcParams['axes.facecolor'] = 'w'

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

def sync_plot(df, metrics, conditions, p_vals=None, colors=None, alpha=None, 
    fname_out=None, **kwargs):
    """
    Plot violin plots for each spike statistic in the given dataframe (df) for the given condition.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to be plotted.
    metrics : list
        List of spike statistics to be plotted.
    conditions : str
        Conditions to be plotted.
    markersize : int, optional
        Size of the markers in the swarm plot. Default is 5.
    fname_out : str, optional
        File path for figure to be saved out to.

    Returns
    -------
    None
    """

    # imports
    import matplotlib.patches as mpatches

    regions = df['brain_structure'].unique()
    colors = ['C0', 'C1', 'C2', 'C3'] if colors is None else colors
    alpha = 1.0 if alpha is None else alpha
    ylabels = kwargs.pop('ylabels', [])
    xlabels = kwargs.pop('xlabels', [])
    # colors = ['red','darkorange','blue','purple']

    p_vals = iter(p_vals) if p_vals is not None else None

    # plot violin plots for each spike statistic
    fig, ax = plt.subplots(len(regions), len(metrics), figsize=(10,8), sharex=True)
    plt.xlabel('region')

    for j, region in enumerate(regions):
        # segment data by region
        region_df = df[df['brain_structure'] == region]

        # plot each metric
        for i, metric in enumerate(metrics):

            ps = next(p_vals) if p_vals is not None else None

            plot_connected_scatter(region_df, metric, conditions, ax=ax[i,j], 
                vp_alpha=alpha, p_vals=ps, line_color='black', colors=colors)

            # add legend on final plot only
            if i==len(metrics)-1 and j==len(regions)-1:
                labels = []
                for ic, color in enumerate(colors):
                    labels.append((mpatches.Patch(color=color, alpha=alpha), conditions[ic]))
                ax[i,j].legend(*zip(*labels), prop={'size':10})

            # label figure on edges only
            if j==0:
                ax[i,j].set_ylabel('' if len(ylabels)==0 else ylabels[i], fontweight='bold')
            if i==len(metrics)-1:
                ax[i,j].set_xlabel(region, fontweight='bold')

            # share correct axes
            if j == 0:
                ax[i,j].sharey(ax[i,j+1])


    # save figure
    if not fname_out is None:
        fig.savefig(fname_out)

def plot_connected_scatter(df, metric, conditions, ax, paired=True, vp_alpha=1.0,
                           scatter_jit=.05, scatter_alpha=0.5, p_vals=None,
                           line_color=None, line_alpha=.1, colors=None, **kwargs):
    """Plot connected violin scattter plots.

    Parameters
    ----------
    df: pd.DataFrame
        pandas DataFrame from which to draw data.
    metric : str
        metric to plot (column from dataframe).
    conditions: list of str
        conditions which to split the data based on.
    ax : AxesSubplot
        Subplot to plot onto.
    paired : bool, optional, default: True
        Plot lines connected pairs of points from dist0 and dist1.
    scatter_jit : float, optional, default: .05
        Scatter random jit standard deviation.
    scatter_alpha : float, optional, default: .05
        Transparency of scatter points.
    line_color : str, optional, default: None
        Color of paired lines
    line_alpha : float, optional, default: .1
        Transparency of paired lines.
    colors : list, optional, default: None
        Colors of the two violin and scatter plots.
    **kwargs : optional
        Additional plotting arguments.

    Returns
    -------
    ax0 : AxesSubplot
        Drawn subplot.
    ax1 : AxesSubplot, optional
        Drawn twin subplot.
    """

    # Pop kwargs
    violin_locs = (1,2,4,5)
    colors = ['C0', 'C1', 'C2', 'C3'] if colors is None else colors
    title = kwargs.pop('title', '')
    ylabel = kwargs.pop('ylabel', '')
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)

    # Separate data based on condition
    condition='state'
    g0 = df[df[condition] == conditions[0]][metric]
    g1 = df[df[condition] == conditions[1]][metric]
    g2 = df[df[condition] == conditions[2]][metric]
    g3 = df[df[condition] == conditions[3]][metric]

    # Violinplots
    vp = ax.violinplot([g0,g1,g2,g3], positions=violin_locs, showextrema=False)
    vp = _set_vp_colors(vp, colors, vp_alpha)

    # Scatterplots

    if isinstance(scatter_jit, (tuple, list)):
        scatter_jit0, scatter_jit1, scatter_jit2, scatter_jit3 = scatter_jit
    else:
        scatter_jit0, scatter_jit1, scatter_jit2, scatter_jit3 = scatter_jit, scatter_jit, scatter_jit, scatter_jit

    xs_a = _weight_scatter_points(g0, violin_locs[0], scatter_jit0)
    xs_a = np.abs(xs_a - violin_locs[0]) + violin_locs[0]

    xs_b = _weight_scatter_points(g1, violin_locs[1], scatter_jit1)
    xs_b = np.abs(xs_b - violin_locs[1]) + violin_locs[1]
    xs_b = -(xs_b - violin_locs[1]) + violin_locs[1]

    xs_c = _weight_scatter_points(g2, violin_locs[2], scatter_jit2)
    xs_c = np.abs(xs_c - violin_locs[2]) + violin_locs[2]

    xs_d = _weight_scatter_points(g3, violin_locs[3], scatter_jit3)
    xs_d = np.abs(xs_d - violin_locs[3]) + violin_locs[3]
    xs_d = -(xs_d - violin_locs[3]) + violin_locs[3]

    ax.scatter(xs_a, g0, alpha=scatter_alpha, color='gray', s=10)
    ax.scatter(xs_b, g1, alpha=scatter_alpha, color='gray', s=10)
    ax.scatter(xs_c, g2, alpha=scatter_alpha, color='gray', s=10)
    ax.scatter(xs_d, g3, alpha=scatter_alpha, color='gray', s=10)


    # Create connected lines between averages
    avg0 = df[df[condition] == conditions[0]].groupby('session')[metric].mean().tolist()
    avg1 = df[df[condition] == conditions[1]].groupby('session')[metric].mean().tolist()
    avg2 = df[df[condition] == conditions[2]].groupby('session')[metric].mean().tolist()
    avg3 = df[df[condition] == conditions[3]].groupby('session')[metric].mean().tolist()

    xs_e = _weight_scatter_points(avg0, violin_locs[0], scatter_jit0)
    xs_e = np.abs(xs_e - violin_locs[0]) + violin_locs[0]

    xs_f = _weight_scatter_points(avg1, violin_locs[1], scatter_jit1)
    xs_f = np.abs(xs_f - violin_locs[1]) + violin_locs[1]
    xs_f = -(xs_f - violin_locs[1]) + violin_locs[1]

    xs_g = _weight_scatter_points(avg2, violin_locs[2], scatter_jit2)
    xs_g = np.abs(xs_g - violin_locs[2]) + violin_locs[2]

    xs_h = _weight_scatter_points(avg3, violin_locs[3], scatter_jit3)
    xs_h = np.abs(xs_h - violin_locs[3]) + violin_locs[3]
    xs_h = -(xs_h - violin_locs[3]) + violin_locs[3]

    ax.scatter(xs_e, avg0, alpha=scatter_alpha, color='black', s=50)
    ax.scatter(xs_f, avg1, alpha=scatter_alpha, color='black', s=50)
    ax.scatter(xs_g, avg2, alpha=scatter_alpha, color='black', s=50)
    ax.scatter(xs_h, avg3, alpha=scatter_alpha, color='black', s=50)

    if paired:

        line_color = 'C0' if line_color is None else line_color

        for i, (d0, d1) in enumerate(zip(avg0, avg1)):
            ax.plot([xs_e[i], xs_f[i]], [d0, d1], color=line_color, alpha=line_alpha)

        for i, (d2, d3) in enumerate(zip(avg2, avg3)):
            ax.plot([xs_g[i], xs_h[i]], [d2, d3], color=line_color, alpha=line_alpha)

    # Significance values
    if p_vals is not None:
        # Get the y-axis limits
        bottom, top = ax.get_ylim()
        y_range = top - bottom

        # Plot the bar
        bar_height = top - (y_range * 0.35)
        bar_tips = bar_height - (y_range * 0.02)

        for i, p in enumerate(p_vals):

            sig_symbol = get_significance(p)

            if sig_symbol is not None:            

                ax.plot([violin_locs[0 + i*2], violin_locs[0 + i*2], violin_locs[1 + i*2], violin_locs[1 + i*2]],
                    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
         
                text_height = bar_height + (y_range * 0.01)
                ax.text((violin_locs[0 + i*2] + violin_locs[1 + i*2]) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')



    # Axis settings
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks((1.5, 4.5))
    ax.set_xticklabels(['behavior', 'presentation'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax


def _weight_scatter_points(dist, loc, std, bins=500):
    """Weight scatter points by density."""

    weights, edges = np.histogram(dist, bins=bins)
    weights = weights / weights.max()

    bins = np.array([edges[:-1], edges[1:]]).T
    bins[-1][1] += 1

    inds = np.zeros(len(dist), dtype=int)
    for i, r in enumerate(dist):
        inds[i] = np.where((r >= bins[:, 0]) & (r < bins[:, 1]))[0][0]

    weights = weights[inds]

    xs = np.zeros_like(dist)

    for i in range(len(dist)):
        xs[i] = np.random.normal(loc, weights[i]*std)

    return xs


def _set_vp_colors(vp, colors, alpha):
    """Update violin plot colors."""


    for i, body in enumerate(vp['bodies']):

        # vp['cmins'].set_color(colors[i])
        # vp['cmaxes'].set_color(colors[i])
        # vp['cbars'].set_color(colors[i])

        vp['bodies'][i].set_color(colors[i])
        vp['bodies'][i].set_facecolor(colors[i])
        vp['bodies'][i].set_edgecolor(colors[i])
        vp['bodies'][i].set_alpha(alpha)

        b = vp['bodies'][i]

        if i%2==0:
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf,
                                                  np.mean(b.get_paths()[0].vertices[:, 0]))
        else:
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],
                                                  np.mean(b.get_paths()[0].vertices[:, 0]),
                                                  np.inf)

    return vp


def get_significance(p):

    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return None


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


def error_scatter_plot(df, x, y, group, colors=None, show=False, **kwargs):
    """
    Plot an errorbar scatter plot grouped by region, aggregated over groups.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing data to be aggregated and plotted.
    x : array-like
        name of independent variable to be plotted.
    y : array-like
        name of dependent variable to be plotted.
    group : str
        group variable to separate and aggregate data by.
    Returns
    -------
    None

    """

    regions = df['brain_structure'].unique()

    # create figure
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    fig.patch.set_facecolor('white') # set background color to white for text legibility
    colors = ['dodgerblue', 'orange'] if colors is None else colors
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')

    for r, region in enumerate(regions):

        region_df = df[df['brain_structure']==region]

        df_mean = region_df.groupby('session')[[x,y]].mean()
        df_std = region_df.groupby('session')[[x,y]].std()
        xlims = (region_df[x].quantile(0.05), region_df[x].quantile(0.95))

        ax.errorbar(df_mean[x].to_numpy(), df_mean[y].to_numpy(), xerr=df_std[x], 
            yerr=df_std[y], color=colors[r], alpha=0.3, label=region, linestyle='', marker='.')

        plot_regression_line(region_df[x], region_df[y], ax=ax, label=region, 
            text_height=0.9-r*0.2, color=colors[r], xlims=xlims)

    ax.legend()
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    if show:
        plt.show()


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


def plot_regression_line(x, y, ax, print_stats=True, text_height=0.9, label='', 
                         xlims=None, color=None):
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
    label : str, optional
        test label to use on regression report

    Returns
    -------
    None
    """

    # imports
    from scipy.stats import linregress

    xlims = (np.nanmin(x), np.nanmax(x)) if xlims is None else xlims

    results = linregress(x, y)
    t_lin = np.linspace(xlims[0], xlims[1], 100)
    lin = results.slope * t_lin + results.intercept
    ax.plot(t_lin, lin, linewidth=5, color='black')
    if color:
        ax.plot(t_lin, lin, linewidth=3, color=color)

    # add regression results text
    if print_stats:
        if results.pvalue < 0.001:
            pval = f"{results.pvalue:.2e}"
        else:
            pval = f"{results.pvalue:.3f}"
        plt.text(1.05, text_height, 
                f"Regression {label}\n" +
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


def plot_time_resolved_params(df, session, window, fs, colors=None, alpha=1.0, title=None, **kwargs):
    """
    Plot normalized time resolved aperiodic parameters.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing lfp parameter data.
    session: str
        session from which to draw data.
    window: tuple of ints
        window indices corresponding to the window of data to plot.
    fs: int
        sampling frequency of data.
    title : str, optional
        title of the produced plot

    Returns
    -------
    None
    """
    colors = ['C0', 'C1', 'C2', 'C3'] if colors is None else colors
    xlabel = kwargs.pop('xlabel', '')
    ylabel = kwargs.pop('ylabel', '')

    sdf = df[df['session'] == session]
    wdf = sdf[(sdf['window_idx'] >= window[0]) & (sdf['window_idx'] < window[1])]
    t = np.linspace(window[0]/fs, window[1]/fs, (window[1]-window[0]))

    fig, ax = plt.subplots(figsize=(8,4))
    plt.set_cmap('Blues')
    for i, series in enumerate(['avg_pupil_area', 'inst_spike_rate', 'exponent', 'offset']):
        ax.plot(t, np.array((wdf[series] - wdf[series].mean())/wdf[series].std()) + (3-i)*4, alpha=alpha, color=colors[i], label=series)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set_title('subject ' + str(session))
    ax.legend(prop={'size':8})

    plt.show()

