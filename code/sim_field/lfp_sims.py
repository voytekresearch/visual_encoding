"""Template script for running analysis across a group of EEG subjects, after pre-processing.
Notes:
-
-
"""

import seaborn as sns
import matplotlib.pyplot as plt
from neurodsp.spectral import compute_spectrum
from neurodsp.plts.spectral import plot_power_spectra
from fooof import FOOOF
import sys
sys.path.append('./')
from funcs import *

###################################################################################################
###################################################################################################

## SETTINGS
fs = 1000

###################################################################################################
###################################################################################################

def main():

    # Initialize any output variables to save out
    LFP_E, LFP_I, t = sim_field(4)
    LFP = LFP_E + LFP_I

    fig1, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax = sns.lineplot(x = t[:5000], y = LFP_E[:5000], ax = axes[0]) # used arbitrary length of 5000 to save time
    sns.lineplot(x = t[:5000], y = LFP_I[:5000], ax = axes[0])
    ax.set_xlim(0,0.2)
    ax.legend(labels=["Excitatory","Inhibitory"])

    ax = sns.lineplot(x = t[:5000], y = LFP[:5000], color='black', ax = axes[1]) # used arbitrary length of 5000 to save time
    ax.set_xlim(0,0.2)
    ax.legend(labels=["LFP"])

    # PSD
    # PSD Excitatory
    freq_e, psd_e = compute_spectrum(LFP_E, fs, method='welch', avg_type='median', nperseg=fs*2)

    # PSD Inhibitory
    freq_i, psd_i = compute_spectrum(LFP_I, fs, method='welch', avg_type='median', nperseg=fs*2)

    # PSD LFP
    freq_lfp, psd_lfp = compute_spectrum(LFP, fs, method='welch', avg_type='median', nperseg=fs*2)

    # Plot the power spectra
    plot_power_spectra([freq_lfp[:200], freq_e[:200], freq_i[:200]],
                    [psd_lfp[:200], psd_e[:200], psd_i[:200]],
                    ['LFP', 'Excitatory', 'Inhibitory'])
    plt.savefig('plot2.png')

    #############################

    LFP_E, LFP_I, t = sim_field(2) # EI ratio = 1 : 2
    LFP2 = LFP_E + LFP_I
    LFP_E, LFP_I, _ = sim_field(6) # EI ratio = 1 : 6
    LFP6 = LFP_E + LFP_I
    # PSD Excitatory
    freq_2, psd_2 = compute_spectrum(LFP2, fs, method='welch', avg_type='median', nperseg=fs*2)
    # PSD Inhibitory
    freq_6, psd_6 = compute_spectrum(LFP6, fs, method='welch', avg_type='median', nperseg=fs*2)
    # Plot the power spectra
    plot_power_spectra([freq_2[:1000], freq_6[:1000]],
                    [psd_2[:1000], psd_6[:1000]],
                    ['EI ratio = 1:2', 'EI ratio = 1:6'])
    plt.savefig('plot3.png')

    # Save any group level files
    fig1.savefig('plot1.png')

    


if __name__ == "__main__":
    main()