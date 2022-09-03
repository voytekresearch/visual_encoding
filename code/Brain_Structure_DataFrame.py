# settings
PROJECT_PATH = 'C:/users/micha/visual_encoding' # 'C:\\Users\\User\\visual_encoding'
SESSION_TYPE = 'functional_connectivity'
BRAIN_STRUCTURES = ['VISp', 'LGd']

# imports
import sys
sys.path.append(PROJECT_PATH)
from allen_vc.utils import get_unit_info

def main():

	# identify manifest path
	manifest_path = PROJECT_PATH + '/data/manifest_files/manifest.json'

	# get unit info for each region of interest
	for structure in BRAIN_STRUCTURES:
		print(f'Getting data for: \t{structure}')
		unit_info = get_unit_info(manifest_path, brain_structure=structure, session_type=SESSION_TYPE)

		# save to file (csv)
		unit_info.to_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv',index=False)

# Can be ran as script
if __name__ == "__main__":
	main()