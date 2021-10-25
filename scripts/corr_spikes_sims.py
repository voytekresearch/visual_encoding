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
from funcs import sim_spikes_general_2stoch
from funcs import sim_homogeneous_pool
from utils import plot_coincidences
from utils import plot_correlations
import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
###################################################################################################

# SETTINGS
N_DT_CC = 20000 # number of time bins for cross correlation
DT = 0.1 # time increment in ms
N_NEURONS = 5 # number of neurons

N_SECONDS = 3600
FS = 1000
ALPHA = 800000.
TAU_C = 1.0E-2
CHUNK_SIZE = 5

# Set paths
BASE_PATH = "../"
FIGURE_PATH = pjoin(BASE_PATH, "figures/CorrSpikes/")
DATA_PATH = pjoin(BASE_PATH, "data/simulations/")

###################################################################################################
###################################################################################################

def main():
    simulate_spikes_general_doubly_stochastic()
    simulate_spikes_homogeneous_pool()

def simulate_spikes_general_doubly_stochastic():
    spikes = sim_spikes_general_2stoch(n_dt=N_DT_CC, n_neurons = N_NEURONS, \
                        dt = 0.1, tau_c = 10.0, firing_rate = 20.0)
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
    firing_rate = np.zeros((5, N_SECONDS * FS))
    spikes = np.zeros((5, N_SECONDS * FS))
    for i_chunk in range(int(N_SECONDS / CHUNK_SIZE)):
        n_timepoints = CHUNK_SIZE * FS
        firing_rate[:,i_chunk*n_timepoints:(i_chunk+1)*n_timepoints], \
                    spikes[:,i_chunk*n_timepoints:(i_chunk+1)*n_timepoints],_ = \
                    sim_homogeneous_pool(rate=20, fs=FS, n_seconds=CHUNK_SIZE, \
                    alpha = ALPHA, tau_c=TAU_C)
                
    _,_, rand_process = sim_homogeneous_pool(rate=20, fs=FS, n_seconds=CHUNK_SIZE, \
                    alpha = ALPHA, tau_c=TAU_C)
    plt.plot(rand_process[:1000], linewidth=0.5)
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_random_process_plot.png'))
    plt.close('all')

    # Plot instantaneous firing rates
    plt.plot(firing_rate.T[:1000,:], linewidth=0.5)
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_firing_rate_plot.png'))
    plt.close('all')

    # Plot correlations of firing rates
    plot_correlations(firing_rate[:,:5000], maxlags=20, plot_model=False)
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_firing_rate_correlations_plot.png'))
    plt.close('all')

    # Plot coincidences
    plot_coincidences(spikes, maxlags = int(TAU_C * FS * 2))
    plt.savefig(pjoin(FIGURE_PATH,'homogeneous_pool_coincidences_plot.png'))
    plt.close('all')

if __name__ == "__main__":
    main()
