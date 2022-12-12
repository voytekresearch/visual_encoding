import os
import numpy as np
import pandas as pd

#Settings
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
BRAIN_STRUCTURES = ['VISp', 'LGd']
STIMULUS_NAME = 'spontaneous'
EPOCH_LENGTH = 10
EPOCHS_FILEPATH = f'{PROJECT_PATH}/data/behavior/running/'+\
f'{STIMULUS_NAME}_random_running_epoch_{EPOCH_LENGTH}s.npz'
STATES = ['stationary', 'running']

# Import custom functions
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import calculate_spike_metrics

def main():
	#Initialize space for data storage
	data = [0]*9

	epochs = np.load(EPOCHS_FILEPATH)

	for structure in BRAIN_STRUCTURES:
		meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

		# Iterate over each session
		for session_id in meta.get('ecephys_session_id').unique():
			spike_matrix = pd.read_pickle(f'{PROJECT_PATH}/data/spike_data'+\
				f'/{str(session_id)}_{structure}_spike_times.pkl')

			# Change spike data format
			raw_spikes = []
			for train in spike_matrix:
				raw_spikes.append(spike_matrix[train])

			# Calculate spike metrics for both states
			for epoch_type in STATES:
				epoch = epochs[f'{session_id}_{epoch_type}']
				if len(epoch)>0:
					#Calculate/combine metrics for storage
					metrics = list(calculate_spike_metrics(raw_spikes, epoch))
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
	
	pd.DataFrame(data=data, columns=labels).\
	to_csv(f'{PROJECT_PATH}/data/synchrony_data/{BRAIN_STRUCTURES}_'+\
		f'{STIMULUS_NAME}_{EPOCH_LENGTH}s_epoch_synchronydf.csv', index=False)

if __name__ == '__main__':
	main()







	