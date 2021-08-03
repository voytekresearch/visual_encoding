"""TODO: Add description"""
import numpy as np
from scipy import signal
from scipy import stats
from neurodsp.spectral import compute_spectrum
from fooof import FOOOF

###################################################################################################
###################################################################################################

def syn_kernel (t, tau):
    # given a specific synaptic kernel type and time constant, this returns a
    # time series of the kernel that spans the time defined (t) in seconds
    #
    # t: time vector in seconds (e.g. t=0:0.001:5)
    # tau: t_decay or [t_rise t_decay] in seconds
    #
    # return: kernel -- the synaptic kernel
    if len(tau) != 2:
        print('Need two time constants for double exponential.')
        return np.array([])
    tpeak = tau[1]*tau[0] / (tau[1]-tau[0]) * np.log(tau[1]/tau[0])
    normf = 1/(-np.exp(-tpeak/tau[0]) + np.exp(-tpeak/tau[1])) # the normalization factor
    kernel = normf * (-np.exp(-t/tau[0])+np.exp(-t/tau[1]))
    return kernel

def pois_spikes (sim_t, dt, N_neu, FR):
    # simulate population spiking of N neurons firing at FR each, return a
    # single spike train that is the total spiking

    # mu parameter for exponential distribution
    MU = 1. / (N_neu*FR)

    # draw ISI from exp RV 
    ISI = np.random.exponential(MU, int((sim_t+2)/MU))
    spk_times = np.cumsum(ISI)
    spk_times = spk_times[spk_times<=sim_t] # potentially inefficient

    # discretize
    bins = np.arange(0, sim_t, dt) + dt/2 # make discretizing bins
    discretized, _ = np.histogram(spk_times, bins = bins, density = False)
    return discretized

def sim_field (EI_ratio, t = 2 * 60, FR_E = 2, FR_I = 5, N_E = 8000, N_I = 2000, tk = 1,
             AMPA_tau = np.array([0.1, 2.])/1000., GABA_tau = np.array([0.5, 10.])/1000.,
             Vr = -65, Ee = 0, Ei = -80, dt=0.001):
    """ FR_E = 2 Firing Rate -- Excitatory
        FR_I = 5 Firing Rate -- Inhibitory
        N_E = 8000 Population -- Excitatory
        N_I = 2000Population -- Inhibitory
        Vr = -65  Resting Membrane Potential
        Ee = 0  AMPA Reversal Potential -- Excitatory
        Ei = -80  GABA_A Reversal Potential -- Inhibitory
        AMPA_tau = np.array([0.1, 2.])/1000.  AMPA Conductance Rise, Decay Time in Seconds
        GABA_tau = np.array([0.5, 10.])/1000.  GABA_A Conductance Rise, Decay Time in Seconds
    """

    fs = 1/dt # sampling rate
    tk = np.arange(0, tk, dt) #PSC kernel time vector
    t = np.arange(0, t, dt) #simulation time vector
    kA = syn_kernel(tk,AMPA_tau)
    kG = syn_kernel(tk,GABA_tau)
    boost = EI_ratio / ((N_I*FR_I*sum(kG))/(N_E*FR_E*sum(kA)))
    spk_E = pois_spikes(t[-1]+tk[-1]+dt, dt, N_E, FR_E)
    spk_I = pois_spikes(t[-1]+tk[-1]+dt, dt, N_I, FR_I)
    GE = np.convolve(spk_E, kA, 'valid') # Total Excitatory Conductance
    GI = np.convolve(spk_I, kG, 'valid') * boost # Total Inhibitory Conductance
    LFP_E = signal.detrend(GE, type = 'constant') * (Ee-Vr) # high-pass drift removal * potential difference
    LFP_I = signal.detrend(GI, type = 'constant') * (Ei-Vr) # high-pass drift removal * potential difference
    return LFP_E, LFP_I, t

def batchsim_PSDs (EI_ratios= np.arange(2, 6.01, 0.2), num_trs = 5, t = 2 * 60, FR_E = 2, FR_I = 5, N_E = 8000, N_I = 2000, tk = 1,
             AMPA_tau = np.array([0.1, 2.])/1000., GABA_tau = np.array([0.5, 10.])/1000.,
             Vr = -65, Ee = 0, Ei = -80, dt=0.001):
    """ Simulate PSD multiple times with an array of different EI_Ratios
        num_trs = 5 Number of trials for each EI ratio
        FR_E = 2 Firing Rate -- Excitatory
        FR_I = 5 Firing Rate -- Inhibitory
        N_E = 8000 Population -- Excitatory
        N_I = 2000Population -- Inhibitory
        Vr = -65  Resting Membrane Potential
        Ee = 0  AMPA Reversal Potential -- Excitatory
        Ei = -80  GABA_A Reversal Potential -- Inhibitory
        AMPA_tau = np.array([0.1, 2.])/1000.  AMPA Conductance Rise, Decay Time in Seconds
        GABA_tau = np.array([0.5, 10.])/1000.  GABA_A Conductance Rise, Decay Time in Seconds
    """
    fs = int(1/dt) # sampling rate
    PSDs = np.zeros([int(fs/2 + 1), len(EI_ratios), num_trs])
    for i in range(len(EI_ratios)):
        for tr in range(num_trs):
            LFP_E, LFP_I, _ = sim_field(EI_ratios[i], t = t, FR_E = FR_E, FR_I = FR_I, N_E = N_E, N_I = N_I, tk = tk,
                                AMPA_tau = AMPA_tau, GABA_tau = GABA_tau, Vr = Vr, Ee = Ee, Ei = Ei, dt=dt)
            LFP = LFP_E + LFP_I
            freq_lfp, psd_lfp = compute_spectrum(LFP, fs, method='welch', avg_type='median', nperseg=fs, noverlap=int(fs/2))
            PSDs[:,i,tr] = psd_lfp
    return PSDs, freq_lfp

def batchfit_PSDs(PSDs, freq_lfp, EI_ratios = np.arange(2, 6.01, 0.2), num_trs = 5, freq_range = [30, 50]):
    slopes = np.zeros([len(EI_ratios), num_trs])
    for i in range(len(EI_ratios)):
        for tr in range(num_trs):
            psd_lfp = PSDs[:,i,tr]
            fm = FOOOF(peak_width_limits=[2, 8], aperiodic_mode='fixed')
            fm.fit(freq_lfp, psd_lfp, freq_range)
            slopes[i, tr] = -fm.aperiodic_params_[1]
    return slopes

def batchcorr_PSDs(PSDs, freq_lfp, EI_ratios = np.arange(2, 6.01, 0.2), center_freqs = np.arange(20, 160.1, 5), 
                    win_len = 20, num_trs = 5):
    rhos = np.zeros([len(center_freqs), num_trs])
    for f in range(len(center_freqs)):
        freq_range = [center_freqs[f]-win_len/2, center_freqs[f]+win_len/2]
        slopes = batchfit_PSDs(PSDs, freq_lfp, EI_ratios, num_trs=num_trs, freq_range=freq_range)
        for tr in range(num_trs):
            rhos[f,tr] = stats.spearmanr(1./EI_ratios, slopes[:,tr]).correlation
    return rhos

def sim_lfp(ei_ratio, n_seconds=2*60, fs=1000, n_neurons=[8000, 2000], 
            firing_rate=[2, 5], tau_r=[0.0001, 0.0005], tau_d=[0.002, 0.01]):
      
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
    
    # imports 
    from neurodsp.sim.aperiodic import sim_synaptic_current
    from neurodsp.sim.transients import sim_synaptic_kernel

    # simulate excitatory and inhibitory currents
    lfp_e = sim_synaptic_current(n_seconds, fs, n_neurons=n_neurons[0], 
                                 firing_rate=firing_rate[0], tau_r=tau_r[0], 
                                 tau_d=tau_d[0], t_ker=None)
    lfp_i = sim_synaptic_current(n_seconds, fs, n_neurons=n_neurons[1], 
                                 firing_rate=firing_rate[1], tau_r=tau_r[1], 
                                 tau_d=tau_d[1], t_ker=None)
    
    # compute desired E:I ratio  
    kernel_e = sim_synaptic_kernel(5*tau_d[0], fs, tau_r[0], tau_d[0])
    kernel_i = sim_synaptic_kernel(5*tau_d[1], fs, tau_r[1], tau_d[1])
    boost = ei_ratio / ((n_neurons[1]*firing_rate[1]*sum(kernel_i)) / 
                        (n_neurons[0] * firing_rate[0] * sum(kernel_e)))
    lfp_i = lfp_i * boost

    # compute lfp
    lfp = lfp_e + lfp_i
    
    return lfp, lfp_e, lfp_i