import os
import numpy as np
import pandas as pd

#Settings
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC = f'{PROJECT_PATH}/data'
BRAIN_STRUCTURES = ['VISp', 'LGd']
epoch_length = 30

import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import calculate_spike_metrics

def main():
	#Initialize space for data storage
	data = [0]*9

	for structure in BRAIN_STRUCTURES:
		meta=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')

		for session_id in meta.get('ecephys_session_id').unique():
			both_epochs = np.load(f'{DATA_LOC}\\epoch_data\\{session_id}_{epoch_length}s_random_epochs.npz')
			spike_matrix = pd.read_pickle(f"{DATA_LOC}/spike_data/{str(session_id)}_{structure}_spike_times.pkl")

			raw_spikes = []
			for train in spike_matrix:
				raw_spikes.append(spike_matrix[train])

			for epoch_type in ['stationary', 'running']:
				if len(both_epochs[epoch_type])>0:
					#Calculate/combine metrics for storage
					epoch = both_epochs[epoch_type]
					metrics = list(calculate_spike_metrics(raw_spikes, epoch))
					metrics+=[epoch, epoch_type, structure, session_id]
					data = np.vstack((data, metrics))
				else:
					metrics = [None]*6 + [epoch_type, structure, session_id]
					data = np.vstack((data, metrics))

	#Save DataFrame including all metrics
	data = np.delete(data, (0), axis=0)
	labels = ['mean_firing_rate', 'coefficient_of_variation', 'SPIKE-distance','SPIKE-synchrony','correlation coefficient', 'epoch', 'state', 'brain_structure','session_id']
	pd.DataFrame(data=data, columns=labels).to_csv(f'{DATA_LOC}/synchrony_data/{BRAIN_STRUCTURES}_{epoch_length}s_epochs_synchronydf.csv', index=False)

if __name__ == '__main__':
	main()







	