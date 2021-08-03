"""Template script for running analysis across a group of EEG subjects, after pre-processing.
Notes:
-
-
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neurodsp.spectral import compute_spectrum
from neurodsp.plts.spectral import plot_power_spectra
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
    
    # simulate LFP for a given E:I ratio
    simulate_lfp()
    
    # simulate LFPs for a range of E:I ratio
    # and correlate E:I ratio with resulting PSD slope
    PSDs, freq = corr_EIRatio_and_slope()
    
    # correlate E:I ratio and slope across a range of fitting freq. ranges
    assess_corr_across_freqs(PSDs, freq)
    
def simulate_lfp():
    
    # Initialize any output variables to save out
    LFP_E, LFP_I, t = sim_field(EI_RATIO_1C)
    LFP = LFP_E + LFP_I

    # Compute power spectrum of LFP and E/I conductance
    _, psd_e = compute_spectrum(LFP_E, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))
    _, psd_i = compute_spectrum(LFP_I, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))
    freq, psd_lfp = compute_spectrum(LFP, FS, method='welch', avg_type='median', nperseg=FS, noverlap=int(FS/2))

    # Figure 1, C. 
    # Plot time-series 
    samples_2_plot = int(np.floor(FS)) # ~1 seconds
    fig_1c, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax = sns.lineplot(x=t[:samples_2_plot], y=LFP_E[:samples_2_plot], ax = axes[0])
    sns.lineplot(x=t[:samples_2_plot], y=LFP_I[:samples_2_plot], ax = axes[0])
    ax.legend(labels=["Excitatory","Inhibitory"])

    ax = sns.lineplot(x=t[:samples_2_plot], y=LFP[:samples_2_plot], ax=axes[1], color='black')
    ax.legend(labels=["LFP"])
    fig_1c.savefig('1C.png')

    # Plot Figure 1, D. 
    # Plot power spectra
    plot_power_spectra([freq, freq, freq], [psd_lfp, psd_e, psd_i],
                    ['LFP', 'Excitatory', 'Inhibitory'])
    plt.savefig('1D.png')
    
    
def corr_EIRatio_and_slope():
    # analysis
    PSDs, freq = batchsim_PSDs(EI_ratios=EI_RATIO, num_trs=N_SIMS)
    slopes = batchfit_PSDs(PSDs, freq, EI_ratios = EI_RATIO, num_trs=N_SIMS, freq_range=F_RANGE_FIT)
    
    # Figure 1, E.
    # Plot power spectra for two different E:I ratios
    psd_0 = np.squeeze(PSDs[:, EI_RATIO==EI_RATIO_1E[0]].mean(axis=2))
    psd_1 = np.squeeze(PSDs[:, EI_RATIO==EI_RATIO_1E[1]].mean(axis=2))
    plot_power_spectra([freq, freq], [psd_0, psd_1],
                       ['EI ratio = 1:%d' %EI_RATIO_1E[0], 
                        'EI ratio = 1:%d' %EI_RATIO_1E[1]]) 
    plt.savefig('1E.png')
    plt.close('all')
    
    # Figure 1, F.
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
    
    return PSDs, freq
 
    
def assess_corr_across_freqs(PSDs, freq):
    rhos = batchcorr_PSDs(PSDs, freq, EI_ratios = EI_RATIO, center_freqs=F_RANGE_CENTER, 
               win_len=F_RANGE_WIDTH, num_trs=N_SIMS)  
     
    # Figure 1, G.
    df2 = pd.DataFrame(rhos, columns = ['Trial1','Trial2','Trial3','Trial4','Trial5'])
    df2['CenterFreq'] = F_RANGE_CENTER
    df2_plot = df2.melt('CenterFreq')
    ax2 = sns.lineplot(data=df2_plot, x = 'CenterFreq', y = 'value', marker='o', color = 'black')
    ax2.set_xlabel('Fit Center Frequency (+-10 Hz)')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation - Fitting Window Plot')
    fig5 = ax2.get_figure()
    fig5.savefig('1G.png')


if __name__ == "__main__":
    main()