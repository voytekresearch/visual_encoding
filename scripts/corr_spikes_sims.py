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
import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
###################################################################################################

# SETTINGS
N_DT_CC = 20000 # number of time bins for cross correlation
DT = 0.1 # time increment in ms
N_NEURONS = 5 # number of neurons

# Set paths
BASE_PATH = "../"
FIGURE_PATH = pjoin(BASE_PATH, "figures/CorrSpikes/")
DATA_PATH = pjoin(BASE_PATH, "data/simulations/")

###################################################################################################
###################################################################################################

def main():
    simulate_spikes_general_doubly_stochastic()

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

if __name__ == "__main__":
    main()
