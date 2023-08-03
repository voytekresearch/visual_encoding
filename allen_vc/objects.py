"""
This file contains the objects used in the analysis of the Allen Brain Institute
visual coding data. The objects are: 
* SessionResults: can be used to load and store results of a single session


"""
# imports
import numpy as np
import pandas as pd
import neo

# define class
class SessionResults:
    def __init__(self, path, session_id, stim_code, load_psd=False, load_tfr=False):
        # display progress
        print(f"\nLoading results for session {session_id}:")

        # save inputs
        self.path = path
        self.session_id = session_id
        self.stim_code = stim_code

        # load psd results
        if load_psd:
            self.load_psd()
        
        try:
            self.psd_params = pd.read_csv(f"{path}/data/lfp_data/lfp_params/{stim_code}/psd/by_session/params_{session_id}.csv")
            print(f"    PSD spectral parameters loaded...")
        except:
            self.psd_params = None
            print(f"    PSD spectral parameters not found...")

        # load tfr results
        if load_tfr:
            self.load_tfr()

        try:
            self.tfr_params = pd.read_csv(f"{path}/data/lfp_data/lfp_params/{stim_code}/tfr/by_session/params_{session_id}.csv")
            print(f"    TFR spectral parameters loaded...")
        except:
            self.tfr_params = None
            print(f"    TFR spectral parameters not found...")
        
        # load spike results
        try:
            all_spike_stats = pd.read_csv(f"{path}/data/spike_stats/{stim_code}.csv")
            self.spike_stats = all_spike_stats.loc[all_spike_stats['session'] == session_id]
            print(f"    Spike stats loaded...")
        except:
            self.spike_stats = None
            print(f"    Spike stats not found...")

        # print summary
        print("    Complete!\n")

    def load_tfr(self):
        try:
            data_in = np.load(f"{self.path}/data/lfp_data/lfp_tfr/{self.stim_code}/spectra_{self.session_id}.npz")
            self.tfr = data_in['tfr']
            self.tfr_freq = data_in['freq']
            print(f"    TFR data loaded...")
        except:
            print(f"TFR data not found...")

    def load_psd(self):
        try:
            data_in = np.load(f"{self.path}/data/lfp_data/lfp_psd/{self.stim_code}/spectra_{self.session_id}.npz")
            self.psd = data_in['spectra']
            self.psd_freq = data_in['freq']
            print(f"    PSD data loaded...")
        except:
            print(f"PSD data not found...")

    def import_block(self, include_lfp=False):
        if include_lfp:
            block = neo.io.NeoMatlabIO(filename=f"{self.path}/data/blocks/lfp/{self.stim_code}/block_{self.session_id}.mat").read()[0]
        else:
            block = neo.io.NeoMatlabIO(filename=f"{self.path}/data/blocks/segmented/{self.stim_code}/block_{self.session_id}.mat").read()[0]

        return block
    