"""Template script for running analysis across a group of EEG subjects, after pre-processing.
Notes:
-
-
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from neurodsp.spectral import compute_spectrum
from neurodsp.plts.spectral import plot_power_spectra
from fooof import FOOOF
import sys
sys.path.append('./')
from funcs import *

###################################################################################################
###################################################################################################

## SETTINGS

# simulation parameters
FS = 1000 # sampling frequency
EI_RATIO = np.linspace(2, 6, 21)
EI_RATIO_1C = 4 # must be in EI_RATIO
EI_RATIO_1E = [2, 6] # must be in EI_RATIO
N_SIMS = 5 # number f replications for each E:I ratio

# spectral parameterization settings
F_RANGE_FIT = [30, 50] # frequency range for slope fit (for batch of EI ratios)
F_RANGE_CENTER = np.arange(20, 160.1, 5) # center of freq range (for batch fitting)
F_RANGE_WIDTH = 20 # width of freq range (for batch fitting)


###################################################################################################
###################################################################################################

def main():

    # Initialize any output variables to save out
    LFP_E, LFP_I, t = sim_field(EI_RATIO_1C)
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
    freq_e, psd_e = compute_spectrum(LFP_E, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))

    # PSD Inhibitory
    freq_i, psd_i = compute_spectrum(LFP_I, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))

    # PSD LFP
    freq_lfp, psd_lfp = compute_spectrum(LFP, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))

    # Plot the power spectra
    plot_power_spectra([freq_lfp[:200], freq_e[:200], freq_i[:200]],
                    [psd_lfp[:200], psd_e[:200], psd_i[:200]],
                    ['LFP', 'Excitatory', 'Inhibitory'])
    # fig2 = plt.figure()
    plt.savefig('1D.png')
    plt.figure().clear()
    plt.cla()
    plt.clf()
    

    #############################

    PSDs, freq_lfp = batchsim_PSDs(EI_ratios=EI_RATIO, num_trs=N_SIMS)
    slopes = batchfit_PSDs(PSDs, freq_lfp, EI_ratios = EI_RATIO, num_trs=N_SIMS, freq_range=F_RANGE_FIT)
    rhos = batchcorr_PSDs(PSDs, freq_lfp, EI_ratios = EI_RATIO, center_freqs=F_RANGE_CENTER, 
                    win_len=F_RANGE_WIDTH, num_trs=N_SIMS)

    df = pd.DataFrame(slopes, columns = ['Trial1','Trial2','Trial3','Trial4','Trial5'])
    df['EIRatio'] = 1./EI_RATIO
    df_plot = df.melt('EIRatio')
    ax = sns.lineplot(data=df_plot, x = 'EIRatio', y = 'value', marker='o', color = 'black')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('g_E : g_I Ratio')
    ax.set_ylabel('Slope (30-50Hz)')
    ax.set_title('EI Ratio, PSD Slope Correlation Plot')
    fig4 = ax.get_figure()
    fig4.savefig('1F.png')
    ax.clear()
    df2 = pd.DataFrame(rhos, columns = ['Trial1','Trial2','Trial3','Trial4','Trial5'])
    df2['CenterFreq'] = F_RANGE_CENTER
    df2_plot = df2.melt('CenterFreq')
    ax2 = sns.lineplot(data=df2_plot, x = 'CenterFreq', y = 'value', marker='o', color = 'black')
    ax2.set_xlabel('Fit Center Frequency (+-10 Hz)')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation - Fitting Window Plot')
    fig5 = ax2.get_figure()
    fig5.savefig('1G.png')

    LFP_E, LFP_I, t = sim_field(EI_RATIO_1E[0]) # EI ratio = 1 : 2
    LFP2 = LFP_E + LFP_I
    LFP_E, LFP_I, _ = sim_field(EI_RATIO_1E[1]) # EI ratio = 1 : 6
    LFP6 = LFP_E + LFP_I
    # PSD Excitatory
    freq_2, psd_2 = compute_spectrum(LFP2, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))
    # PSD Inhibitory
    freq_6, psd_6 = compute_spectrum(LFP6, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))
    # Plot the power spectra
    plot_power_spectra([freq_2[:1000], freq_6[:1000]],
                    [psd_2[:1000], psd_6[:1000]],
                    ['EI ratio = 1:2', 'EI ratio = 1:6'])
    plt.savefig('1E.png')

    #############################

    # Save any group level files
    fig1.savefig('1C.png')

    


if __name__ == "__main__":
    main()