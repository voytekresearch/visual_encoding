# imports
import os
import pandas as pd
import pickle

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand

# settings
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
BRAIN_STRUCTURES = ['VISp','LGd']

import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_spiking_data

def main():

    # Define/create directories
    dir_results = f'{PROJECT_PATH}/data/spike_data'
    if not os.path.exists(dir_results): 
        os.mkdir(dir_results)
    
    # Create Allensdk cache object
    manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"

    for structure in BRAIN_STRUCTURES:
        print(f'Analyzing Brain Structure: {structure}')
        meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

        for session_id in meta.get('ecephys_session_id').unique():
            print(f'Analyzing session: \t{session_id}')
            spike_times, spike_amplitudes, mean_waveforms = get_spiking_data(session_id, manifest_path, structure)

            # create a binary pickle file 
            for variable, var_str in zip([spike_times,spike_amplitudes,mean_waveforms],
                                          ['spike_times', 'spike_amplitudes',
                                            'mean_waveforms']):
                fname_out = f"{dir_results}/{str(session_id)}_{structure}_{var_str}.pkl"
                h = open(fname_out,"wb")
                pickle.dump(variable, h)
                h.close()


if __name__ == '__main__':
    main()