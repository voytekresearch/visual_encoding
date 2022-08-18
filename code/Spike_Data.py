# imports
import os
import numpy as np
import pandas as pd
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand

# settings
PROJECT_PATH = 'C:/users/micha/visual_encoding' # 'C:\\Users\\User\\visual_encoding'
BRAIN_STRUCTURE = 'VISp'

def main():

    # Define/create directories
    dir_results = f'{PROJECT_PATH}/data/spike_data'
    if not os.path.exists(dir_results): 
        os.mkdir(dir_results)
    
    # Create Allensdk cache object
    manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # Retrive data and store in dictionaries
    spike_times={}
    spike_amplitudes={}
    mean_waveforms={}

    meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{BRAIN_STRUCTURE}_DataFrame.csv')

    for ses in meta.get('ecephys_session_id').unique():
        session = cache.get_session_data(ses, isi_violations_maximum = np.inf,
                                amplitude_cutoff_maximum = np.inf,
                                presence_ratio_minimum = -np.inf)
        for unit in meta[meta.get('ecephys_session_id')==ses].get('unit_id').unique():
            spike_times[unit]=session.spike_times[unit]
            spike_amplitudes[unit]=session.spike_amplitudes[unit]
            mean_waveforms[unit]=session.mean_waveforms[unit]

    # create a binary pickle file 
    for variable, var_str in zip([[spike_times,spike_amplitudes,mean_waveforms],
                                  ['spike_times', 'spike_amplitudes',
                                   'mean_waveforms']]):
        h = open(f"{dir_results}/{BRAIN_STRUCTURE}_{var_str}.pkl","wb")
        pickle.dump(variable, h)
        h.close()


if __name__ == '__main__':
    main()