import os
import numpy as np
import pandas as pd
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand
#The resulting DataFrame contains all units within the given brain structure from both the functional_connectivity and brain_observatory dataset

brain_structure_acronym='VISp'

def main():

    #Define data directory and create Allensdk cache object
    data_directory = 'C:\\Users\\User\\visual_encoding\\data\\manifest_files'
    manifest_path = os.path.join(data_directory, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # Retrive data and store in dictionaries
    spike_times={}
    spike_amplitudes={}
    mean_waveforms={}

    meta=pd.read_csv(f'C:\\Users\\User\\visual_encoding\\data\\brain_structure_DataFrames\\{brain_structure_acronym}_DataFrame.csv')

    for ses in meta.get('ecephys_session_id').unique():
        session = cache.get_session_data(ses, isi_violations_maximum = np.inf,
                                amplitude_cutoff_maximum = np.inf,
                                presence_ratio_minimum = -np.inf)
        for unit in meta[meta.get('ecephys_session_id')==ses].get('unit_id').unique():
            spike_times[unit]=session.spike_times[unit]
            spike_amplitudes[unit]=session.spike_amplitudes[unit]
            mean_waveforms[unit]=session.mean_waveforms[unit]

    # create a binary pickle file 
    f = open(f"C:\\Users\\User\\visual_encoding\\data\\pickles/{brain_structure_acronym}_spike_times.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(spike_times,f)

    # close file
    f.close()

    g = open(f"C:\\Users\\User\\visual_encoding\\data\\pickles/{brain_structure_acronym}_spike_amplitudes.pkl","wb")
    pickle.dump(spike_amplitudes,g)
    g.close()

    h = open(f"C:\\Users\\User\\visual_encoding\\data\\pickles/{brain_structure_acronym}_mean_waveforms.pkl","wb")
    pickle.dump(mean_waveforms,h)
    h.close()

if __name__ == '__main__':
    main()