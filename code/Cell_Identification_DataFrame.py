import os
import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#NOTE: Must run Brain_Structure_DataFrame and Spike_Time_Alignment_DataFrame for given brain_structure beforehand

brain_structure_acronym='VISp'
stimulus_name_duration_level=['fast_pulses', 1,4]

#Define function that takes two series as an argument and returns an array of booleans that denote whether or not the value in the second series is >= twice the value in the first
def is_cre(baseline,evoked):
    result=np.array([])
    b=np.array(baseline)
    e=np.array(evoked)
    i=0
    for value in e:
        if value==0:
            result=np.append(result,False)
        elif 2*b[i]<=value:
            result=np.append(result,True)
        else:
            result=np.append(result,False)
        i+=1
    return result

def main():
	#Define data directory and create Allensdk cache object
	data_directory = 'C:\\Users\\User\\visual_encoding\\data\\manifest_files'
	manifest_path = os.path.join(data_directory, "manifest.json")
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	#Apply cre id function
	unit_spike_rate_df=pd.read_csv(f'C:\\Users\\User\\visual_encoding\\data\\brain_structure_DataFrames\\{brain_structure_acronym}_with_spikes_{stimulus_name_duration_level[0]}_{str(stimulus_name_duration_level[1])}ms_{stimulus_name_duration_level[2]}level.csv')
	unit_spike_rate_df=unit_spike_rate_df.assign(cre_positive=is_cre(unit_spike_rate_df.get('baseline_rate'),unit_spike_rate_df.get('evoked_rate')))

	unit_spike_rate_df.to_csv(f'C:\\Users\\User\\visual_encoding\\data\\brain_structure_DataFrames\\{brain_structure_acronym}_cre_identification_{stimulus_name_duration_level[0]}_{str(stimulus_name_duration_level[1])}ms_{stimulus_name_duration_level[2]}level.csv',index=False)

	#Print the proportion of cre postive units within each genotype for each specimen
	for specimen in unit_spike_rate_df.get('specimen_id').unique():
	    specimen_df=unit_spike_rate_df[unit_spike_rate_df.get('specimen_id')==specimen]
	    print(f'specimen: {specimen}')
	    for genotype in specimen_df.get('genotype').unique():
	        genotype_df=specimen_df[specimen_df.get('genotype')==genotype]
	        num_cre=genotype_df[genotype_df.get('cre_positive')==1].shape[0]
	        print(f'\t{genotype}: {num_cre}/{genotype_df.shape[0]}')
	    print('')

if __name__ == '__main__':
	main()