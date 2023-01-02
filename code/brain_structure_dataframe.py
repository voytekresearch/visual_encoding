"""
Load and save DataFrames including information on all units across BRAIN_STRUCTURES.
"""

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc" # Allen manifest.json
PROJECT_PATH = "G:/Shared drives/visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding
RELATIVE_PATH_OUT = "data/brain_structure_DataFrames" # where to save output relative to both paths above

# settings - dataset details
SESSION_TYPE = 'functional_connectivity'
BRAIN_STRUCTURES = ['VISp', 'LGd']

# imports
import sys
import os
sys.path.append(REPO_PATH)
from allen_vc.utils import get_unit_info

def main():

	# Define/create directories
	for base_path in [PROJECT_PATH, MANIFEST_PATH]:
		dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
		if not os.path.exists(dir_results): 
			os.makedirs(dir_results)

	# identify manifest path
	manifest_path = f'{MANIFEST_PATH}/manifest_files/manifest.json'

	# get unit info for each region of interest
	for structure in BRAIN_STRUCTURES:
		print(f'Getting data for: \t{structure}')
		unit_info = get_unit_info(manifest_path, brain_structure=structure, \
			session_type=SESSION_TYPE)

		# save to file (csv)
		for base_path in [PROJECT_PATH, MANIFEST_PATH]:
			unit_info.to_csv(f'{base_path}/{RELATIVE_PATH_OUT}/{structure}_DataFrame.csv',\
				index=False)

# Can be ran as script
if __name__ == "__main__":
	main()