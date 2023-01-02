"""
Retrieve spiking data for all sessions with given BRAIN_STRUCTURES.
"""

# imports
import os
import pandas as pd
import pickle

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc"
PROJECT_PATH = "G:/Shared drives/visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding
RELATIVE_PATH_OUT = "data/spike_data/spike_times"

# settings - dataset details
BRAIN_STRUCTURES = ['VISp','LGd']

import sys
sys.path.append(REPO_PATH)
from allen_vc.utils import gen_neo_spiketrains

def main():

    # Define/create directories
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # Create Allensdk cache object
    manifest_path = f"{MANIFEST_PATH}/manifest_files/manifest.json"

    for structure in BRAIN_STRUCTURES:
        print(f'Analyzing Brain Structure: {structure}')
        meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

        # loop through sessions
        for session_id in meta.get('ecephys_session_id').unique():
            print(f'Analyzing session: \t{session_id}')
            
            # Get spiking data
            spiketrains = gen_neo_spiketrains(session_id, manifest_path, structure)

            # save to file
            for base_path in [PROJECT_PATH, MANIFEST_PATH]:
                fname_out = f"{base_path}/{RELATIVE_PATH_OUT}/{str(session_id)}_{structure}.pkl"
                h = open(fname_out,"wb")
                pickle.dump(spiketrains, h)
                h.close()


if __name__ == '__main__':
    main()