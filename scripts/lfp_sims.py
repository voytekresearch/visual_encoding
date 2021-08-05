"""Template script for running analysis across a group of EEG subjects, after pre-processing.
Notes:
- sim_field is deprecated, sim_lfp is prefered
- the generated figures are under figures/EISlopes/
- the saved intermediate data are under data/simulations/
"""

## IMPORTS
import sys
sys.path.append('../sim_field')
from os.path import join as pjoin
from funcs import sim_field, batchsim_PSDs, batchfit_PSDs, batchcorr_PSDs
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neurodsp.spectral import compute_spectrum
from neurodsp.plts.spectral import plot_power_spectra


###################################################################################################
###################################################################################################

# SETTINGS

# simulation parameters
FS = 1000  # sampling frequency
EI_RATIO = np.linspace(2, 6, 21)
EI_RATIO_1C = 4  # must be in EI_RATIO
EI_RATIO_1E = [2, 6]  # must be in EI_RATIO
N_SIMS = 5  # number f replications for each E:I ratio

# spectral parameterization settings
# frequency range for slope fit (for batch of EI ratios)
F_RANGE_FIT = [30, 50]
# center of freq range (for batch fitting)
F_RANGE_CENTER = np.arange(20, 160.1, 5)
F_RANGE_WIDTH = 20  # width of freq range (for batch fitting)

# Set paths for the project
BASE_PATH = "../"
FIGURE_PATH = pjoin(BASE_PATH, "figures/EISlope/")
DATA_PATH = pjoin(BASE_PATH, "data/simulations/")

###################################################################################################
###################################################################################################


def main():

    # simulate LFP for a given E:I ratio
    simulate_lfp()

    # simulate LFPs for a range of E:I ratio
    # and correlate E:I ratio with resulting PSD slope
    psd_batch, freq = corr_EIRatio_and_slope()

    # correlate E:I ratio and slope across a range of fitting freq. ranges
    assess_corr_across_freqs(psd_batch, freq)


def simulate_lfp():

    # Initialize any output variables to save out
    lfp_e, lfp_i, t = sim_field(EI_RATIO_1C)
    lfp = lfp_e + lfp_i

    # Compute power spectrum of LFP and E/I conductance
    _, psd_e = compute_spectrum(lfp_e, FS, method='welch', avg_type='median',
                                nperseg=FS, noverlap=int(FS/2))
    _, psd_i = compute_spectrum(lfp_i, FS, method='welch', avg_type='median',
                                nperseg=FS, noverlap=int(FS/2))
    freq, psd = compute_spectrum(lfp, FS, method='welch', avg_type='median',
                                 nperseg=FS, noverlap=int(FS/2))

    # Figure 1, C.
    # Plot time-series
    samples_2_plot = int(np.floor(FS))  # ~1 seconds
    fig_1c, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax = sns.lineplot(x=t[:samples_2_plot], y=lfp_e[:samples_2_plot],
                      ax=axes[0])
    sns.lineplot(x=t[:samples_2_plot], y=lfp_i[:samples_2_plot], ax=axes[0])
    ax.legend(labels=["Excitatory", "Inhibitory"])

    ax = sns.lineplot(x=t[:samples_2_plot], y=lfp[:samples_2_plot], ax=axes[1],
                      color='black')
    ax.legend(labels=["LFP"])
    fig_1c.savefig(pjoin(FIGURE_PATH,'1C.png'))

    # Plot Figure 1, D.
    # Plot power spectra
    plot_power_spectra([freq, freq, freq], [psd, psd_e, psd_i],
                       ['LFP', 'Excitatory', 'Inhibitory'])
    plt.savefig(pjoin(FIGURE_PATH,'1D.png'))


def corr_EIRatio_and_slope():
    # analysis
    psd_batch, freq = batchsim_PSDs(ei_ratios=EI_RATIO, num_trs=N_SIMS)
    slopes = batchfit_PSDs(psd_batch, freq,
                           freq_range=F_RANGE_FIT)

    # Figure 1, E.
    # Plot power spectra for two different E:I ratios
    psd_0 = np.squeeze(psd_batch[:, EI_RATIO == EI_RATIO_1E[0], 0])
    psd_1 = np.squeeze(psd_batch[:, EI_RATIO == EI_RATIO_1E[1], 0])
    plot_power_spectra([freq, freq], [psd_0/sum(psd_0), psd_1/sum(psd_1)],
                       ['EI ratio = 1:%d' % EI_RATIO_1E[0],
                        'EI ratio = 1:%d' % EI_RATIO_1E[1]])
    plt.savefig(pjoin(FIGURE_PATH,'1E.png'))
    plt.close('all')

    # Figure 1, F.
    # Plot Figure 1, F.
    df = pd.DataFrame(slopes,
                      columns=['Trial1', 'Trial2', 'Trial3', 'Trial4', 'Trial5'])
    df['EIRatio'] = EI_RATIO
    df_plot = df.melt('EIRatio')
    ax = sns.lineplot(data=df_plot, x='EIRatio', y='value', marker='o',
                      color='black')
    ax.set_xlabel('g_E : g_I Ratio')
    ax.set_ylabel('Slope (30-50Hz)')
    ax.set_title('EI Ratio, PSD Slope Correlation Plot')
    ax.set_xlim(EI_RATIO[-1], EI_RATIO[0])
    ax.set_xticks([EI_RATIO_1E[0], EI_RATIO_1C, EI_RATIO_1E[1]])
    ax.set_xticklabels(['1:%.0f' % EI_RATIO_1E[0], '1:%.0f' % EI_RATIO_1C,
                        '1:%.0f' % EI_RATIO_1E[1]])
    fig_1f = ax.get_figure()
    fig_1f.savefig(pjoin(FIGURE_PATH,'1F.png'))
    plt.close('all')

    return psd_batch, freq


def assess_corr_across_freqs(psd_batch, freq):

    # Correlate EI ratio and slope fit in different frequency ranges
    rhos = batchcorr_PSDs(psd_batch, freq, ei_ratios=EI_RATIO,
                          center_freqs=F_RANGE_CENTER,
                          win_len=F_RANGE_WIDTH, num_trs=N_SIMS)

    # Plot Figure 1, G.
    df2 = pd.DataFrame(rhos,
                       columns=['Trial1', 'Trial2', 'Trial3', 'Trial4', 'Trial5'])
    df2['CenterFreq'] = F_RANGE_CENTER
    df2_plot = df2.melt('CenterFreq')
    ax2 = sns.lineplot(data=df2_plot, x='CenterFreq', y='value', marker='o',
                       color='black')
    ax2.set_xlabel('Fit Center Frequency (+-10 Hz)')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation - Fitting Window Plot')
    fig_1g = ax2.get_figure()
    fig_1g.savefig(pjoin(FIGURE_PATH,'1G.png'))
    plt.close('all')


if __name__ == "__main__":
    main()
