import os
import numpy as np
import pandas as pd
from allen_vc.utils import calculate_spike_metrics
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH = 'C:\\Users\\User\\visual_encoding' #'C:/users/micha/visual_encoding'
DATA_LOC = f'{PROJECT_PATH}/data'
BRAIN_STRUCTURES = ['VISp', 'LGd']
epoch_length = 30

def main():
	#Retrieve appropriate session ids
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()
	session_ids = sessions[sessions.get('session_type')=='functional_connectivity'].index

	#Initialize space for data storage
	data = np.array()

	for structure in BRAIN_STRUCTURES:
		for session_id in session_ids:
			#Load stored epoch and spike data
			both_epochs = np.load(f'{DATA_LOC}\\epoch_data\\{session_id}_{epoch_length}s_random_epoch.npz')

			spike_matrix = pd.read_pickle(f"{DATA_LOC}/spike_data/{str(session_id)}_{structure}_spike_times.pkl")

			for epoch_type in ['stationary', 'running']:
				#Calculate/combine metrics for storage
				epoch = both_epochs[epoch_type]
				metrics = calculate_spike_metrics(spike_matrix, epoch)
				metrics.append([epoch, epoch_type, structure])
				np.vstack(data, metrics)

	#Save DataFrame including all metrics
	labels = ['mean_firing_rate', 'coefficient_of_variation', 'SPIKE-distance','SPIKE-synchrony','correlation coefficient', 'epoch', 'state', 'brain_structure']
	pd.DataFrame(data=data, columns=labels).to_csv(f'{DATA_LOC}/synchrony_data/{BRAIN_STRUCTURES}_{epoch_length}s_epochs_synchronydf.csv', index=False)

if __name__ == '__main__':
	main()







	