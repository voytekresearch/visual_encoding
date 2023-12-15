# Visual Encoding
*Michael Preston, Sawyer Figueroa, Simon Fei*

------------------------------------------------
## Abstract

Electrical brain activity can be measured at various temporal and spatial scales. For instance, mesoscopic signals such as electroencephalography (EEG) and local field potentials (LFP) reflect activity across populations of neurons. Notably oscillations in EEG and LFP have been associated with mechanisms of memory and cognition. However, emerging research has shown that non-oscillatory, aperiodic activity also serves as a biomarker of disease, age, cortical region, and cognitive state. Although the significance of aperiodic activity is strongly supported, the underlying mechanisms and physiological origin have not been fully characterized. In this study, we leveraged an open dataset (Allen Institute for Brain Science: Visual Coding - Neuropixels dataset) to investigate the relationship between population spiking activity and aperiodic LFP activity. Here we show that aperiodic LFP activity indexes the rate and synchrony of population spiking activity within and between cognitive and behavioral states.  Our current results support previous findings that broadband LFP power reflects the firing rate of a population. Specifically, we found a strong negative correlation between rate and low frequency power with a concomitant strong positive correlation at high frequency ranges, indicative of a singular, aperiodic process. Surprisingly, spike synchrony was found to be negatively correlated with the aperiodic exponent of the LFP, contrary to the predictions of previous models i.e. greater spike synchrony was associated with flatter power spectra. These findings support the idea that aperiodic EEG and LFP activity is a physiologically meaningful signal, providing information about the underlying population spiking statistics.  Further investigation into the aperiodic components of electrical brain waves has the potential to provide valuable information surrounding cognition not present in oscillatory activity. More specifically, characterizing the physiological origin of aperiodic activity will advance our understanding of its functional role in cognition and disease.

------------------------------------------------
## Dataset

Allen Institute MindScope Program (2019). Allen Brain Observatory -- Neuropixels Visual Coding [dataset]. Available from brain-map.org/explore/circuits. Siegle JH, Jia X, Durand S, et al. 
Survey of spiking in the mouse visual system reveals functional hierarchy. Nature. Published online January 20, 2021. doi:10.1038/s41586-020-03171-x

For an exploratory data analysis of the dataset, check `notebooks/explore_dataset`.

------------------------------------------------
## Use

Our data organization and analysis methodology can be be performed via the following commands:

Clone the respository
`git clone https://github.com/voytekresearch/visual_encoding.git`
Travel to the main directory
`cd visual_encoding`
Run the code!
1. `python code/step1_create_session_blocks.py`
2. `python code/step2_create_spontaneous_segments.py`
3. `python code/step2_create_stimulus_segments.py`
4. `python code/step3_add_lfp_to_stimulus_segments.py`

\* Note Python version 3.7 is required along with installation of the AllenSDK API

-----------------------------------------------
## Society for Neuroscience (2023)

Figures, methods, and materials relevant to our SfN 2023 poster can be found in `notebooks/sfn_2023`.

----------------------------------------------
## References
1. Buzsáki G, Anastassiou CA, Koch C. The origin of extracellular elds and currents-EEG, ECoG, LFP and spikes. Nat Rev Neurosci. 2012;13(6):407-420. doi:10.1038/nrn3241
2. Miller KJ, Sorensen LB, Ojemann JG, Den Nijs M. Power-law scaling in the brain surface electric potential. PLoS Comput Biol. 2009;5(12). doi:10.1371/journal.pcbi.1000609
3. Gao R, Peterson EJ, Voytek B. Inferring synaptic excitation/inhibition balance from eld potentials. NeuroImage. 2017;158:70-78. doi:10.1016/j.neuroimage.2017.06.078
4. Kelly RC, Smith MA, Kass RE, Lee TS. Local eld potentials indicate network state and account for neuronal response variability. J Comput Neurosci. 2010;29(3):567-579. doi:10.1007/s10827-009-0208-9
5. Poulet JFA, Crochet S. The Cortical States of Wakefulness. Front Syst Neurosci. 2019;12:64. doi:10.3389/fnsys.2018.00064
6. Dataset: Allen Institute MindScope Program (2019). Allen Brain Observatory -- Neuropixels Visual Coding [dataset]. Available from brain-map.org/explore/circuits. Siegle JH, Jia X, Durand S, et al. Survey of spiking in the mouse visual system reveals functional hierarchy. Nature. Published online January 20, 2021. doi:10.1038/s41586-020-03171-x
7. Jun JJ, Steinmetz NA, Siegle JH, et al. Fully integrated silicon probes for high-density recording of neural activity. Nature. 2017;551(7679):232-236. doi:10.1038/nature24636
8. Siegle JH, Jia X, Durand S, et al. Survey of spiking in the mouse visual system reveals functional hierarchy. Nature. 2021;592(7852):86-92. doi:10.1038/s41586-020-03171-x
9. Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F. Monitoring spike train synchrony. J Neurophysiol. 2013;109(5):1457-1472. doi:10.1152/jn.00873.2012
10. Kreuz T, Mulansky M, Bozanic N. SPIKY: a graphical user interface for monitoring spike train synchrony. J Neurophysiol. 2015;113(9):3432-3445. doi:10.1152/jn.00848.2014
11. Donoghue T, Haller M, Peterson EJ, et al. Parameterizing neural power spectra into periodic and aperiodic components. Nat Neurosci. 2020;23(12):1655-1665. doi:10.1038/s41593-020-00744-x
