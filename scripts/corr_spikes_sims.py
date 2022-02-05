"""Template script for simulating and analyzing correlated spike trains
Notes:
- run by 'python corr_spikes_sims.py'
- the generated figures are under figures/CorrSpikes/
- the saved intermediate data are under data/simulations/
"""

## IMPORTS

import sys
sys.path.append('../sim_field')
from os.path import join as pjoin
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from plots import plot_coincidences,  plot_correlations
from funcs import sim_spikes_general_2stoch, sim_homogeneous_pool, \
                  get_correlation_matrices, gen_spikes_mixture

###################################################################################################
###################################################################################################

# SETTINGS
N_DT_CC = 20000 # number of time bins for cross correlation
DT = 0.1 # time increment in ms
N_NEURONS = 5 # number of neurons

N_SECONDS = 2000
FS = 1000
ALPHA = 300. # I don't know why ALPHA needs to be this high for the functions to work
TAU_C = 1.0E-2
CHUNK_SIZE = 5
FIRING_RATE = 20

# Set paths
BASE_PATH = "C:/Users/micha/visual_encoding"
FIGURE_PATH = pjoin(BASE_PATH, "figures/CorrSpikes/")
DATA_PATH = pjoin(BASE_PATH, "data/simulations/")

###################################################################################################
###################################################################################################

def main():
    # simulate_spikes_general_doubly_stochastic()
    # simulate_spikes_homogeneous_pool()
    simulate_gaussian_mixture()

def simulate_spikes_general_doubly_stochastic():
    spikes = sim_spikes_general_2stoch(n_dt=N_DT_CC, n_neurons = N_NEURONS, \
                        dt = 0.1, tau_c = 10.0, firing_rate = FIRING_RATE)
    times = np.arange(0,N_DT_CC*DT,DT)

    # Plot spike roster
    plt.eventplot([times[spikes.T[i]==1] for i in range(N_NEURONS)])
    plt.savefig(pjoin(FIGURE_PATH,'general_doubly_stochastic_roster_plot.png'))
    plt.close('all')
    # Plot emperical cross correlation
    plt.acorr(times[spikes.T[0]==1][:32] - times[spikes.T[1]==1][:32], maxlags=20, lw=10)
    plt.savefig(pjoin(FIGURE_PATH,'general_doubly_stochastic_empirical_cc_plot.png'))
    plt.close('all')

def simulate_spikes_homogeneous_pool():
    spikes, rand_process = sim_homogeneous_pool(mu=FIRING_RATE, fs=FS, n_seconds=N_SECONDS, \
                         variance = 300, tau_c=TAU_C)

    # Plot random process
    plt.plot(rand_process[:1000], linewidth=0.5)
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_random_process_plot.png'))
    plt.close('all')

    # # # Plot correlations of firing rates
    # plot_correlations(rand_process[:2000], maxlags=20, plot_model=False)
    # plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_rand_process_autocorrelation_plot.png'))
    # plt.close('all')

    # Plot rate autocorr
    rand_process_mean_removed = signal.detrend(rand_process, type='constant')
    plt.acorr(rand_process_mean_removed[:5000], maxlags = 20)
    plt.savefig(pjoin(FIGURE_PATH,'random_walk_rate_autocorr.png'))
    plt.close('all')

    # Plot coincidences
    plot_coincidences(spikes, maxlags = int(TAU_C * FS * 2) * 3, plot_model = True)
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_coincidences_plot.png'))
    plt.close('all')

def simulate_gaussian_mixture():
    covariances, firing_rates_array = get_correlation_matrices(N_NEURONS, FIRING_RATE, .1, 0)

    # Plot spike roster
    # print(covariances)
    plt.pcolor(covariances)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('covariance matrix')
    plt.savefig(pjoin(FIGURE_PATH,'gaussian_mixture_covariance_matrix.png'))
    plt.close('all')

    # run simualation
    spikes, inst_firing_rates, _ = gen_spikes_mixture(N_SECONDS, covariances, firing_rates_array, FS, TAU_C)
    
    # Plot instantaneous firing rates
    # detrended_rates = np.array([inst_firing_rates[a,:5000] for a in range(len(inst_firing_rates))])
    plt.plot(inst_firing_rates.T[:1000,:], linewidth=0.5)
    # plt.plot(detrended_rates.T)
    plt.savefig(pjoin(FIGURE_PATH,'gaussian_mixture_firing_rate_plot.png'))
    plt.close('all')

    # Plot correlations of firing rates
    detrended_rates = np.array([signal.detrend(inst_firing_rates[a,:5000]) for a in range(len(inst_firing_rates))])
    plot_correlations(detrended_rates, maxlags=20, plot_model=True)
    plt.savefig(pjoin(FIGURE_PATH,'gaussian_mixture_firing_rate_correlations_plot.png'))
    plt.close('all')

    # Plot coincidences
    plot_coincidences(spikes, maxlags = int(TAU_C * FS * 2) * 3)
    plt.savefig(pjoin(FIGURE_PATH,'gaussian_mixture_coincidences_plot.png'))
    plt.close('all')

    _,_, rand_processes = gen_spikes_mixture(CHUNK_SIZE, covariances, firing_rates_array, FS, TAU_C, ALPHA)
    plt.plot(rand_processes.T[:1000,:], linewidth=0.5)
    plt.savefig(pjoin(FIGURE_PATH,'gaussian_mixture_rand_processes_plot.png'))
    plt.close('all')


if __name__ == "__main__":
    main()
