"""Functions to simulate LFPs, compute PSDs of the LFPs, and fit slopes onto the PSDs"""

import numpy as np
from scipy import signal
from scipy import stats
from scipy import linalg
from neurodsp.spectral import compute_spectrum
from neurodsp.sim.aperiodic import sim_synaptic_current
from neurodsp.sim.transients import sim_synaptic_kernel
from aperiodic import sim_random_walk
# from neurodsp.sim import sim_random_walk
from fooof import FOOOFGroup

##########################################################################
##########################################################################


def syn_kernel(n_seconds, tau):
    """ *Deprecated: check neurodsp.sim.transients.sim_synaptic_kernel*
    given a specific synaptic kernel type and time constant, this returns a
    time series of the kernel that spans the time defined (n_seconds) in seconds

    Parameters
    ----------
    n_seconds : 1d array, (e.g. n_seconds=0:0.001:5)
        time vector in seconds
    tau : 1x2 array, [t_rise t_decay]
        Time constants of synaptic kernel, in seconds. (rise fall)

    Returns
    -------
    kernel : 1d array
        the kernel of the synaptic current

    Examples
    --------
    >>> t_ker = np.arange(0, t_ker, dt)
    >>> tau_exc = np.array([0.1, 2.])/1000.
    >>> kernel_e = syn_kernel(t_ker,tau_exc)
    """

    if len(tau) != 2:
        print('Need two time constants for double exponential.')
        return np.array([])
    tpeak = tau[1] * tau[0] / (tau[1] - tau[0]) * np.log(tau[1] / tau[0])
    # the normalization factor
    normf = 1 / (-np.exp(-tpeak / tau[0]) + np.exp(-tpeak / tau[1]))
    kernel = normf * (-np.exp(-n_seconds / tau[0]) + np.exp(-n_seconds / tau[1]))
    return kernel

def pois_spikes(n_seconds, dt, n_neurons, firing_rate):
    """ simulate population spiking of N neurons firing at firing_rate each, return a
    single spike train that is the total spiking

    Parameters
    ----------
    n_seconds : float
        simulation time vector in seconds
    dt : float
        Time increment unit
    n_neurons : int
        number of neurons
    firing_rate : float
        average firing rate of a neuron in Hz

    Returns
    -------
    discretized : 1d array
        the discretized sum of spikes per time bin

    Examples
    --------
    >>> spk_E = pois_spikes(n_seconds=20, dt = 0.001, n_neurons = 8000, firing_rate = 2)
    """

    # mu parameter for exponential distribution
    mu = 1. / (n_neurons * firing_rate)

    # draw ISI from exp RV
    isi = np.random.exponential(mu, int((n_seconds + 2) / mu))
    spk_times = np.cumsum(isi)
    spk_times = spk_times[spk_times <= n_seconds]  # potentially inefficient

    # discretize
    bins = np.arange(0, n_seconds, dt) + dt / 2  # make discretizing bins
    discretized, _ = np.histogram(spk_times, bins=bins, density=False)
    return discretized

def sim_spikes_general_2stoch(n_dt, n_neurons = 5, dt = 0.1, tau_c = 10.0, firing_rate = 20.0):
    """ simulate population spiking of N neurons firing at firing_rate each, return a
    spike trains roster of size (n_dt by n_neurons)

    Parameters
    ----------
    n_dt : int
        number of time bins where a spike could occur
    dt : float
        Time increment unit (in ms)
    tau_c : float
        Time constant TODO: ???
    n_neurons : int
        number of neurons
    firing_rate : float
        average firing rate of a neuron in Hz

    Returns
    -------
    spikes : 2d (n_dt by n_neurons) array
        the spike roster of n_neurons where 0 is no spike at the time bin and 1 is a spike

    Examples
    --------
    >>> spk_roster = sim_spikes_general_2stoch(n_dt=2000, dt = 0.1, n_neurons = 5, firing_rate = 20)
    """
    sigma=np.sqrt(1-np.exp(-2.*dt/tau_c)) # standard deviation
    ccvf_f=np.exp(-dt/tau_c); # f(s) part of CCVF # TODO: ???
    # define Y(t) = (yi(t)) a vector of N independent Ornstein-Uhlenbeck processes
    rand_processes = np.random.normal(loc=0.0, scale=1.0, size=(n_dt, n_neurons)) * \
                        ccvf_f + np.random.normal(loc=0.0, scale=sigma, size=(n_dt, n_neurons))
    # C = (ci,j) the matrix of correlations
    correlations = np.zeros((n_neurons,n_neurons))
    for i in range(correlations.shape[0]):
        for j in range(correlations.shape[1]):
            if j < i:
                rand_num=np.random.uniform(low=1.0,high=101.0)
                correlations[i,j] = rand_num
                correlations[j,i] = rand_num
        correlations[i,i] = firing_rate + i # TODO: ???
    # mean firing rate of each neuron
    firing_rates = [correlations[i,i] for i in range(n_neurons)] 
    # Choleskey Decomposition
    corr_rate_squared = np.copy(correlations)
    for i in range(n_neurons):
        corr_rate_squared[i,i] = corr_rate_squared[i,i]**2
    c_factor = linalg.cholesky(corr_rate_squared) # Choleskey Factor L(C = LLT)
    # instantaneus firing rates
    inst_rates = np.zeros(rand_processes.shape)
    for i in range(n_dt):
        inst_rates[i] = firing_rates + c_factor.dot(rand_processes[i])
    get_spikes = np.vectorize(lambda inst_rate : \
                1 if np.random.uniform() < inst_rate*0.1*0.001 else 0)
    spikes = get_spikes(inst_rates)
    return spikes

def sim_homogeneous_pool(n_neurons=5, mu=20, variance=300, n_seconds=1, fs=1000,
                         tau_c=0.01):
    """
    simulate population spiking of N neurons firing at firing_rate each, return a
    spike trains roster of size (n_seconds*fs by n_neurons)

    Parameters
    ----------
    n_neurons : scalar, optional
        number of neurons in population. The default is 5.
    mu : scalar, optional
        mean of random walk. The default is 20.
    sigma : scalar, optional
        std of random walk. The default is 300.
    n_seconds : scalar, optional
        duration of signal (s). The default is 1.
    fs : scalar, optional
        sampling frequency (Hz). The default is 1000.
    tau : scalar, optional
        timescale (sec). The default is 0.01.

    Returns
    -------
    spikes : 2D array
        spike trains.
    rand_process : 1D array
        instantanious firing rate.

    """

    # simulate randon process (Ornstein-Uhlenbeck)
    rand_process, _ = sim_ou_process(n_seconds, fs, mu=mu, sigma=variance**0.5, tau=tau_c)
    # rand_process[rand_process < 0] = 0 # ensure all positive values

    # generate spikes from OU process
    spikes = np.zeros([n_neurons, len(rand_process)])

    # turn rates into spikes
    for j_bin in range(len(rand_process)):
        for i_neuron in range(n_neurons):
            if rand_process[j_bin] / fs > np.random.uniform():
                spikes[i_neuron, j_bin] = 1
    
    return spikes, rand_process

def get_correlation_matrices(n_neurons, firing_rate, xcorr_scalar, firing_rate_std):
    # C = (c_i,j) the matrix of covariances
    # The diagonal would be rates scaled by diagonal coefficient alpha
    covariances = np.zeros((n_neurons,n_neurons))
    for i in range(covariances.shape[0]):
        for j in range(covariances.shape[1]):
            if j < i:
                # rand_num=np.random.uniform(low=1.0,high=101.0)
                rand_num = (np.random.uniform(low=0,high=1) + 1.) * xcorr_scalar
                covariances[i,j] = rand_num
                covariances[j,i] = rand_num

    # mean firing rate of each neuron, C(i,i), diagnal coefficients
    # variances of the processes x_i(t)
    # firing_rates_array = np.array([(1 + .05 * i) * firing_rate for i in range(n_neurons)])
    firing_rates_array = np.random.normal(firing_rate, firing_rate_std, n_neurons) # not sure to use uniform or normal
    firing_rates = np.zeros((n_neurons,n_neurons)) # matrix D with d_i,i = r_i^2
    np.fill_diagonal(firing_rates, firing_rates_array**2)

    # Calcuate diagonal coefficient alpha
    inv_firing_rates = linalg.inv(firing_rates) # D^-1
    eig_values, _ = linalg.eig(inv_firing_rates.dot(covariances))
    diag_coef = -np.real(eig_values).min()


    # Choleskey Decomposition
    # covariances with the diagnal coefficients (firing rate) squared
    covariances = covariances + diag_coef * firing_rates * 1.01

    return covariances, firing_rates_array

def gen_spikes_mixture(n_seconds, covariances, firing_rates_array, fs, tau_c, alpha):
    
    sigma=(2*tau_c*alpha)**0.5
    n_neurons = len(covariances)

    rand_processes = np.zeros((n_neurons, int(n_seconds * fs)))
    for i_neuron in range(n_neurons):
        # rand_processes[i_neuron] = sim_random_walk(n_seconds, fs, theta=1/tau_c, mu=0, sigma=sigma)
        rand_processes[i_neuron], _ = sim_ou_process(n_seconds, fs, tau_c, mu=firing_rates_array[i_neuron], sigma=sigma)

    # compute cholesky
    root_covariances = linalg.cholesky(covariances).T # Choleskey Factor L(C = LLT)

    # dot random process and root correlation
    LY = np.dot(root_covariances, rand_processes)

    # compute firing rate
    # firing_rates = firing_rates_array
    # inst_firing_rates = firing_rates[:, np.newaxis] + LY 
    inst_firing_rates = LY 

    # turn rates into spikes
    spikes = np.zeros((n_neurons, n_seconds * fs))
    for j_bin in range(n_seconds * fs):
        for i_neuron in range(n_neurons):
            if inst_firing_rates[i_neuron, j_bin] / fs > np.random.uniform():
                spikes[i_neuron, j_bin] = 1

    # get_spikes = np.vectorize(lambda inst_rate : \
    #             1 if np.random.uniform() < inst_rate*dt else 0)
    # spikes = get_spikes(inst_firing_rates)
    
    return spikes, inst_firing_rates, rand_processes

def sim_field(ei_ratio, n_seconds=2 * 60, firing_rate_e=2, firing_rate_i=5, n_neurons_e=8000, n_neurons_i=2000, t_ker=1,
              tau_exc=np.array([0.1, 2.]) / 1000., tau_inh=np.array([0.5, 10.]) / 1000.,
              v_rest=-65, e_reversal_e=0, e_reversal_i=-80, dt=0.001):
    """ *Deprecated: use sim_lfp instead for faster speed*
    Simulate LFP using Gao 2017's model

    Parameters
    ----------
    firing_rate_e : float, default: 2
        Firing Rate -- Excitatory
    firing_rate_i: float, default: 5
        Firing Rate -- Inhibitory
    n_neurons_e : int, default: 8000
        Population -- Excitatory
    n_neurons_i : int, default: 2000
        Population -- Inhibitory
    v_rest : float, default: -65
        Resting Membrane Potential
    e_reversal_e : float, default: 0
        AMPA Reversal Potential -- Excitatory
    e_reversal_i : float, default: -80
        GABA_A Reversal Potential -- Inhibitory
    tau_exc : 1x2 array, default: np.array([0.1, 2.])/1000.
        AMPA Conductance Rise, Decay Time in Seconds
    tau_inh : 1x2 array, default: np.array([0.5, 10.])/1000.
        GABA_A Conductance Rise, Decay Time in Seconds
    ei_ratios : 1d array, default: np.arange(2, 6.01, 0.2)
        The EI Ratios to simulate
    t_ker : float, default: 1
        kernel time, in seconds

    Returns
    -------
    lfp_e : 1d array
        the excitatory component of the LFP
    lfp_i : 1d array
        the inhibitory component of the LFP
    times : 1d array
        the time vector of the LFP

    Examples
    --------
    >>> lfp_e, lfp_i, times = sim_field(ei_ratio=4)
    """

    t_ker = np.arange(0, t_ker, dt)  # PSC kernel time vector
    times = np.arange(0, n_seconds, dt)  # simulation time vector
    kernel_e = syn_kernel(t_ker, tau_exc)
    kernel_i = syn_kernel(t_ker, tau_inh)
    boost = ei_ratio / ((n_neurons_i * firing_rate_i* sum(kernel_i)) / (n_neurons_e * firing_rate_e * sum(kernel_e)))
    spk_E = pois_spikes(times[-1] + t_ker[-1] + dt, dt, n_neurons_e, firing_rate_e)
    spk_I = pois_spikes(times[-1] + t_ker[-1] + dt, dt, n_neurons_i, firing_rate_i)
    g_e = np.convolve(spk_E, kernel_e, 'valid')  # Total Excitatory Conductance
    # Total Inhibitory Conductance
    g_i = np.convolve(spk_I, kernel_i, 'valid') * boost
    # high-pass drift removal * potential difference
    lfp_e = signal.detrend(g_e, type='constant') * (e_reversal_e - v_rest)
    # high-pass drift removal * potential difference
    lfp_i = signal.detrend(g_i, type='constant') * (e_reversal_i - v_rest)
    return lfp_e, lfp_i, times, spk_E, spk_I

def sim_lfp_mixture(n_seconds=2 * 60, n_neurons = 50, firing_rate = 20., tau_c = 1.0E-2, alpha = 100000., fs=1000, t_ker=1, tau_r=0.0001, tau_d=0.002, e_reversal = 0., v_rest=-65.):
    covariances, firing_rates_array = get_correlation_matrices(n_neurons, firing_rate, .1, 0)
    kernel = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    spikes, _, _ = gen_spikes_mixture(n_seconds, covariances, firing_rates_array, fs, tau_c, alpha)
    lfps = []
    for i_neuron in range(len(spikes)):
        # lfps[i_neuron, np.newaxis] = signal.detrend(np.convolve(spikes[i_neuron], kernel, 'valid'), type='constant')
        # lfps = np.vstack((lfps, signal.detrend(np.convolve(spikes[i_neuron], kernel, 'valid'), type='constant')))
        lfps.append(signal.detrend(np.convolve(spikes[i_neuron], kernel, 'valid'), type='constant') * (e_reversal - v_rest))
    lfps = np.array(lfps)
    lfp = lfps.sum(axis = 0)
    return lfp, lfps

def sim_lfp_pool(n_seconds=2 * 60, n_neurons = 50, firing_rate = 20., tau_c = 1.0E-2, alpha = 100000., fs=1000, t_ker=1, tau_r=0.0001, tau_d=0.002, e_reversal = 0., v_rest=-65.):
    kernel = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    spikes, _ = sim_homogeneous_pool(n_neurons, firing_rate, n_seconds, fs, alpha, tau_c)
    lfps = []
    for i_neuron in range(len(spikes)):
        lfps.append(signal.detrend(np.convolve(spikes[i_neuron], kernel, 'valid'), type='constant') * (e_reversal - v_rest))
    lfps = np.array(lfps)
    lfp = lfps.sum(axis = 0)
    return lfp, lfps
    
def batchsim_PSDs(ei_ratios=np.arange(2, 6.01, 0.2), num_trs=5, n_seconds=2 * 60,
                    firing_rate=[2, 5], n_neurons=[8000, 2000], t_ker=1, tau_exc=np.array([0.1, 2.]) / 1000.,
                    tau_inh=np.array([0.5, 10.]) / 1000., v_rest=-65, e_reversal=[0, -80], dt=0.001, method='neurodsp'):
    """ Simulate PSD multiple times with an array of different ei_ratios

    Parameters
    ----------
    num_trs : int, default: 5
        Number of trials for each EI ratio
    firing_rate : 1x2 array, default: (2 5)
        Firing rate of neurons in each population. (excitatory inhibitory)
    n_neurons_e : int, default: 8000
        Population -- Excitatory
    n_neurons_i : int, default: 2000
        Population -- Inhibitory
    v_rest : float, default: -65
        Resting Membrane Potential
    e_reversal_e : float, default: 0
        AMPA Reversal Potential -- Excitatory
    e_reversal_i : float, default: -80
        GABA_A Reversal Potential -- Inhibitory
    tau_exc : 1x2 array, default: np.array([0.1, 2.])/1000.
        AMPA Conductance Rise, Decay Time in Seconds
    tau_inh : 1x2 array, default: np.array([0.5, 10.])/1000.
        GABA_A Conductance Rise, Decay Time in Seconds
    ei_ratios : 1d array, default: np.arange(2, 6.01, 0.2)
        The EI Ratios to simulate
    t_ker : float, default: 1
        kernel time, in seconds
    method: string, default: 'neurodsp'
        the method to generate the LFP

    Returns
    -------
    psd_batch : (fs/2+1)xlen(ei_ratios)x(num_trs) 3d array
        Simulated power spectral densities
    freq_lfp : (fs/2+1)x1 array
        Frequency indexes of the power spectral densities

    Examples
    --------
    >>> psd_batch, freq_lfp = batchsim_PSDs(ei_ratios=np.linspace(2, 6, 21), num_trs=5)
    """
    fs = int(1 / dt)  # sampling rate
    psd_batch = np.zeros([int(fs / 2 + 1), len(ei_ratios), num_trs])
    for i in range(len(ei_ratios)):
        for tr in range(num_trs):
            if method == 'neurodsp':
                # simulate lfp
                lfp, _, _ = sim_lfp(ei_ratios[i], n_seconds=n_seconds, fs=fs,
                                    n_neurons=n_neurons, firing_rate=firing_rate,
                                    tau_r=[tau_exc[0], tau_inh[0]],
                                    tau_d=[tau_exc[1], tau_inh[1]], e_reversal=e_reversal)

            else:
                # simulate lfp
                lfp_e, lfp_i, _ = sim_field(ei_ratios[i], n_seconds=n_seconds, firing_rate_e=firing_rate[0], firing_rate_i=firing_rate[1],
                                            n_neurons_e=n_neurons[0], n_neurons_i=n_neurons[1], t_ker=t_ker, tau_exc=tau_exc,
                                            tau_inh=tau_inh, v_rest=v_rest, e_reversal_e=e_reversal[0], e_reversal_i=e_reversal[1], dt=dt)
                lfp = lfp_e + lfp_i

            # compute PSD
            freq_lfp, psd_lfp = compute_spectrum(
                lfp, fs, method='welch', avg_type='median', nperseg=fs, noverlap=int(fs / 2))
            psd_batch[:, i, tr] = psd_lfp

    return psd_batch, freq_lfp

def batchfit_PSDs(psd_batch, freq, freq_range=[30, 50]):
    """Fits slopes that maintains the overall dimensions of PSDs by squeeze and unsqueeze
    the PSDs arrays internally

    Parameters
    ----------
    psd_batch : n dimensional array
        A batch of Power Spectral Density with the last dimension being each PSD array
    freq_lfp : 1d numpy array
        the frequency indexes all PSDs share
    freq_range : 1x2 list, default: [30, 50]
        the frequency range for the slope fit

    Returns
    -------
    slopes : (n - 1) dimensional array
        the shape-matched slopes of the PSDs

    Examples
    --------
    >>> slopes = batchfit_PSDs(psd_batch, freq, freq_range = [30, 50])
    """
    shapeT = psd_batch.T.shape[:-1]  # transpose shape
    # transpose then squeeze psd_batch into a 2D array
    psd_array = psd_batch.T.reshape(np.prod(psd_batch.shape[1:]), len(psd_batch))
    # fake peak_width_limit to supress warnings
    fg = FOOOFGroup(aperiodic_mode='fixed',
                    peak_width_limits=[2, 8], max_n_peaks=0)
    # set n_job = -1 to parallelize
    fg.fit(freq, psd_array, freq_range, n_jobs=-1)
    slopes_array = -fg.get_params('aperiodic_params', 'exponent')
    slopes = slopes_array.reshape(shapeT).T  # unsqueeze the slopes array

    return slopes

def batchcorr_PSDs(psd_batch, freq_lfp, ei_ratios=np.arange(2, 6.01, 0.2),
                    center_freqs=np.arange(20, 165, 5), win_len=20, num_trs=5):
    """Calculate the correlations between the aperiodic exponent slopes and
    EI-Ratios in different time windows for a few trials of PSDs.
    Note: Has a specific use as described above
    The psd_batch should contain the PSDs for different EI-Ratios and different trials

    Parameters
    ----------
    psd_batch : len(ei_ratios)x(num_trs)x(n) 3d array
        A batch of Power Spectral Density with the last dimension being each PSD array
    freq_lfp : 1d numpy array
        the frequency indexes all PSDs share
    ei_ratios : 1d array, default: np.arange(2, 6.01, 0.2)
        The EI Ratios to simulate
    center_freqs : 1d array, default: np.arange(20, 165, 5)
        the center frequencies of the time windows
    win_len : float, default: 20
        the len of each window in Hz
    num_trs : int, default: 5
        Number of trials for each EI ratio

    Returns
    -------
    rhos : len(center_freqs)x(num_trs) 2d array
        the spearman correlations between the aperiodic slopes and ei_ratios for
        each time window for each trial

    Examples
    --------
    >>> rhos = batchcorr_PSDs(psd_batch, freq, ei_ratios = np.linspace(2, 6, 21),
                          center_freqs=np.arange(20, 160.1, 5),
               win_len=20, num_trs=5)
    """
    rhos = np.zeros([len(center_freqs), num_trs])
    for f in range(len(center_freqs)):
        freq_range = [
            center_freqs[f] - win_len / 2,
            center_freqs[f] + win_len / 2]
        slopes = batchfit_PSDs(psd_batch, freq_lfp, freq_range=freq_range)
        for tr in range(num_trs):
            rhos[f, tr] = stats.spearmanr(
                1. / ei_ratios, slopes[:, tr]).correlation
    return rhos

def sim_lfp(ei_ratio, n_seconds=2 * 60, fs=1000, n_neurons=[8000, 2000],
            firing_rate=[2, 5], tau_r=[0.0001, 0.0005], tau_d=[0.002, 0.01],
            t_ker=1, e_reversal=[0, -80]):
    """Simulate LFP using neuroDSP functionality.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_neurons : 1x2 array, default: (8000 2000)
        Number of neurons in each population. (excitatory inhibitory)
    firing_rate : 1x2 array, default: (2 5)
        Firing rate of neurons in each population. (excitatory inhibitory)
    tau_r : 1x2 array, default: (0.0001 0.0005)
        Rise time of synaptic kernel, in seconds. (excitatory inhibitory)
    tau_d : 1x2 array, default: (0.002 0.01)
        Decay time of synaptic kernel, in seconds. (excitatory inhibitory)
    e_reversal : 1x2 array, default: (0 -80)
        reversal potential (excitatory inhibitory)

    Returns
    -------
    lfp : 1d array
        Simulated local field potential
    lfp_e : 1d array
        Simulated local field potential
    lfp_i : 1d array
        Simulated local field potential


    Examples
    --------
    >>> sig = sim_lfp(n_seconds=120, fs=1000)
    """
    # simulate excitatory and inhibitory conductances
    g_e = sim_synaptic_current(n_seconds, fs, n_neurons=n_neurons[0],
                               firing_rate=firing_rate[0], tau_r=tau_r[0],
                               tau_d=tau_d[0], t_ker=t_ker)
    g_i = sim_synaptic_current(n_seconds, fs, n_neurons=n_neurons[1],
                               firing_rate=firing_rate[1], tau_r=tau_r[1],
                               tau_d=tau_d[1], t_ker=t_ker)

    # compute desired E:I ratio
    kernel_e = sim_synaptic_kernel(t_ker, fs, tau_r[0], tau_d[0])
    kernel_i = sim_synaptic_kernel(t_ker, fs, tau_r[1], tau_d[1])
    boost = ei_ratio / ((n_neurons[1] * firing_rate[1] * sum(kernel_e)) /
                        (n_neurons[0] * firing_rate[0] * sum(kernel_i)))
    g_i = g_i * boost

    # detrend conductance time-series
    g_e = signal.detrend(g_e, type='constant')
    g_i = signal.detrend(g_i, type='constant')

    # compute excitatory and inhibitory currents
    lfp_e = g_e * (-65 - e_reversal[0])
    lfp_i = g_i * (-65 - e_reversal[1])

    # compute lfp
    lfp = lfp_e + lfp_i

    return lfp, lfp_e, lfp_i

def sim_ou_process(n_seconds, fs, tau, mu=100., sigma=10.):
    ''' 
    Simulate an Ornstein-Uhlenbeck process with a dymanic timescale.
    
    
    Parameters
    ----------
    n_seconds : float
        Simulation time (s)
    fs : float
        Sampling rate (Hz)
    tau : float
        Timescale of signal (s)
    mu : float, optional, default: 100.
        Mean of signal
    sigma : float, optional, default: 10.
        Standard deviation signal

    Returns
    signal : 1d array
        Simulated Ornstein-Uhlenbeck process
    time : 1d array
        time vector for signal

    References
    ----------
    https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
    
    '''

    # initialize signal and set first value equal to the mean
    signal = np.zeros(int(np.ceil(n_seconds * fs)))
    signal[0] = mu
    
    # define constants in OU equation (to speed computation) 
    dt = 1 / fs
    sqrtdt = np.sqrt(dt)
    rand = np.random.randn(len(signal))
    
    # simulate OU
    for ii in range(len(signal)-1):
        signal[ii + 1] = signal[ii] + \
                        dt * (-(signal[ii] - mu) / tau) + \
                        sigma * np.sqrt(2/tau) * sqrtdt * rand[ii]
    
    # define time vector
    time = np.linspace(0, n_seconds, len(signal))
    
    return signal, time
