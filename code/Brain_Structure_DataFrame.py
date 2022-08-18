# imports
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# settings
PROJECT_PATH = 'C:/users/micha/visual_encoding' # 'C:\\Users\\User\\visual_encoding'
BRAIN_STRUCTURE = 'VISp'

def main():

	#Define data directory and create Allensdk cache object
	data_directory = f'{PROJECT_PATH}\\data\\manifest_files'
	manifest_path = os.path.join(data_directory, "manifest.json")
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	#Get all units and query for VISp units
	units = cache.get_units()
	meta_total=units.reset_index().rename(columns={'id':'unit_id'})
	meta_df=meta_total[meta_total.get('ecephys_structure_acronym')==BRAIN_STRUCTURE].get(['unit_id','ecephys_session_id','ecephys_probe_id','specimen_id']).drop_duplicates()

	#Convert DataFrame to csv
	meta_df.to_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{BRAIN_STRUCTURE}_DataFrame.csv',index=False)

# Can be ran as script
if __name__ == "__main__":
	main()