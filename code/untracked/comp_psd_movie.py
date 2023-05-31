"""
Epoch LFP during natural movie watching

"""

# imports
import os
import numpy as np
# from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from neurodsp.spectral import compute_spectrum
from mne.time_frequency import psd_array_multitaper

# settings
MANIFEST_PATH = "D:/datasets/allen_vc"
PROJECT_PATH = "G:/Shared drives/visual_encoding"
SESSION_TYPE = 'functional_connectivity'

# dataset details
DURATION = 30
FS = 1250 # LFP sampling freq

# ! EXAMPLE DATA FOR TESTING !
SESSION_IDS = [791319847]
PROBE_IDS = [805008604]

def main():

    # Define/create directories
    dir_results_0 = f'{PROJECT_PATH}/data/lfp_psd/natural_movie'
    dir_results_1 = f'{MANIFEST_PATH}/data/lfp_psd/natural_movie'
    for dir_results in [dir_results_0, dir_results_1]:
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # # Create Allensdk cache object
    # print('loading cache')
    # cache = EcephysProjectCache.from_warehouse(manifest=f"{MANIFEST_PATH}/manifest.json")

    # # get session info for dataset of interest
    # print('getting session info')
    # sessions_all = cache.get_session_table()
    # session_ids = sessions_all[sessions_all['session_type']==SESSION_TYPE].index.values

    # for session_id in session_ids: # ! TESTING
    for session_id in SESSION_IDS:
        print(f'loading session: \t{session_id}')
        # # load session data and get probe info
        # session = cache.get_session_data(session_id)
        # probe_ids = session.probes.index.values

        # for probe_id in probe_ids: # ! TESTING
        for probe_id in PROBE_IDS:
            
            # load LFP data
            print('loading spectral results')
            data_in = np.load(f"{MANIFEST_PATH}/data/lfp_epochs/natural_movie/{session_id}_{probe_id}_lfp_movie.npz")
            lfp = data_in['lfp']

            # initialize vars
            print('coputeing PSD')
            # freq, temp = compute_spectrum(lfp[0,0], FS, method='medfilt')
            temp, freq = psd_array_multitaper(lfp[0,0], FS)
            spectra = np.zeros([lfp.shape[0], lfp.shape[1], len(temp)])

            # compute psd
            for i_chan in range(len(lfp)):
                # _, spectra[i_chan] = compute_spectrum(lfp[i_chan], FS, method='medfilt')
                spectra[i_chan], _ = psd_array_multitaper(lfp[i_chan], FS)

            # save results
            for dir_results in [dir_results_0, dir_results_1]:
                print('saving')
                np.savez(f"{dir_results}/{session_id}_{probe_id}_psd_movie.npz", spectra=spectra, freq=freq) 
                print('done saving')


if __name__ == '__main__':
    main()