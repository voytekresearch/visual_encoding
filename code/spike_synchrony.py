"""
Create a dataframe including information about a set of spike synchrony metrics
over a set of loaded epochs.
"""

import os
import numpy as np
import pandas as pd
import pickle
import quantities as pq

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = r"G:\Shared drives\visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding

# settings - data of interest
BRAIN_STRUCTURES = ['VISp', 'LGd']
STIMULUS_NAME = 'spontaneous'
BLOCK = 4
EPOCH_LENGTH = 10
STATES = ['stationary', 'running']

# Import custom functions
import sys
sys.path.append(REPO_PATH)
from allen_vc.utils import calculate_spike_metrics

def main():
	#Initialize space for data storage
	data = [0]*9

	epochs = np.load(f'{PROJECT_PATH}/data/behavior/running/epoch_times/'+\
		f'{STIMULUS_NAME}_{BLOCK}_{EPOCH_LENGTH}s_random.npz')

	for structure in BRAIN_STRUCTURES:
		meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

		# Iterate over each session
		for session_id in meta.get('ecephys_session_id').unique():
			# load Neo SpikeTrains
			spiketrains = pd.read_pickle(f'{PROJECT_PATH}/data/spike_data'+\
				f'/spike_times/{str(session_id)}_{structure}.pkl')

			# Calculate spike metrics for both states
			for epoch_type in STATES:
				epoch = epochs[f'{session_id}_{epoch_type}']

				if len(epoch)>0:
					# epoch spiking data for state
					spiketrains_state = [spiketrain.time_slice(epoch[0]*pq.s, epoch[1]*pq.s) \
					for spiketrain in spiketrains]

					# Calculate/combine metrics for storage
					metrics = list(calculate_spike_metrics(spiketrains_state))
					metrics += [epoch, epoch_type, structure, session_id]
					data = np.vstack((data, metrics))
				else:
					metrics = [None]*6 + [epoch_type, structure, session_id]
					data = np.vstack((data, metrics))

	#Save DataFrame including all metrics
	data = np.delete(data, (0), axis=0)
	labels = ['mean_firing_rate', 'coefficient_of_variation', \
	'SPIKE-distance','SPIKE-synchrony','correlation coefficient', \
	'epoch', 'state', 'brain_structure','session_id']
	
	for base_path in [PROJECT_PATH, MANIFEST_PATH]:
		pd.DataFrame(data=data, columns=labels).\
		to_csv(f'{PROJECT_PATH}/data/synchrony_data/{BRAIN_STRUCTURES}_'+\
			f'{STIMULUS_NAME}_{BLOCK}_{EPOCH_LENGTH}s.csv', index=False)

if __name__ == '__main__':
	main()







	