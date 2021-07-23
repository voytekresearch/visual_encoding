"""TODO: Add description"""
import numpy as np
from scipy import signal

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