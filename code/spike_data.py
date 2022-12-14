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
from allen_vc.utils import gen_neo_spiketrains

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

        # loop through sessions
        for session_id in meta.get('ecephys_session_id').unique():
            print(f'Analyzing session: \t{session_id}')
            
            # Get spiking data
            spiketrains = gen_neo_spiketrains(session_id, manifest_path, structure)

            # save to file
            ...


if __name__ == '__main__':
    main()