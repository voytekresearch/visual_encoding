import os
import numpy as np
import pickle
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#NOTE: Must run Brain_Structure_DataFrame and Spike_Time_Alignment_DataFrame for given brain_structure beforehand

BRAIN_STRUCTURE='VISp'
PROJECT_PATH='C:\\Users\\User\\visual_encoding'
stimulus_name_duration_level=['fast_pulses', 1,2]

#Define function that takes two series as an argument and returns an array of booleans that denote whether or not the value in the second series is >= twice the value in the first
def is_cre(cre_proportions):
    return np.array(cre_proportions)>0.5

def main():
	#Define data directory and create Allensdk cache object
	dir_results=f'{PROJECT_PATH}\\data\\brain_structure_DataFrames'
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	#Apply cre id function
	unit_spike_rate_df=pd.read_csv(f'{dir_results}\\{BRAIN_STRUCTURE}_with_spikes_{stimulus_name_duration_level[0]}_{str(stimulus_name_duration_level[1])}s_{stimulus_name_duration_level[2]}level.csv')
	unit_spike_rate_df=unit_spike_rate_df.assign(cre_positive=is_cre(unit_spike_rate_df.get('units_cre_proportion')))

	unit_spike_rate_df.to_csv(f'{dir_results}\\{BRAIN_STRUCTURE}_cre_identification_{stimulus_name_duration_level[0]}_{str(stimulus_name_duration_level[1])}s_{stimulus_name_duration_level[2]}level.csv',index=False)

	#Print the proportion of cre postive units within each genotype for each specimen and save data to dictionary
	if os.path.exists(f"{PROJECT_PATH}\\data\\pickles\\{BRAIN_STRUCTURE}_cre_ratios"):
		cre_ratios=pd.read_pickle(f"{PROJECT_PATH}\\data\\pickles\\{BRAIN_STRUCTURE}_cre_ratios")
	else:
		cre_ratios={}
	cre_ratios[f'{stimulus_name_duration_level[0]}_{stimulus_name_duration_level[1]}_{stimulus_name_duration_level[2]}']={}
	for specimen in unit_spike_rate_df.get('specimen_id').unique():
	    specimen_df=unit_spike_rate_df[unit_spike_rate_df.get('specimen_id')==specimen]
	    cre_ratios[f'{stimulus_name_duration_level[0]}_{stimulus_name_duration_level[1]}_{stimulus_name_duration_level[2]}'][specimen]={}
	    #print(f'specimen: {specimen}')
	    for genotype in specimen_df.get('genotype').unique():
	        genotype_df=specimen_df[specimen_df.get('genotype')==genotype]
	        num_cre=genotype_df[genotype_df.get('cre_positive')==1].shape[0]
	        cre_ratios[f'{stimulus_name_duration_level[0]}_{stimulus_name_duration_level[1]}_{stimulus_name_duration_level[2]}'][specimen][genotype]=num_cre/genotype_df.shape[0]
	        #print(f'\t{genotype}: {num_cre}/{genotype_df.shape[0]}')
	    #print('')

	f = open(f"{PROJECT_PATH}\\data\\pickles\\{BRAIN_STRUCTURE}_cre_ratios","wb")
	pickle.dump(cre_ratios,f)
	f.close()

if __name__ == '__main__':
	main()