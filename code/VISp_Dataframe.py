import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def main():

	#Define data directory and create Allensdk cache object
	data_directory = 'ManifestFiles'
	manifest_path = os.path.join(data_directory, "manifest.json")
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	#Get all units and query for VISp units
	units = cache.get_units()
	meta_total=units.reset_index().rename(columns={'id':'unit_id'})
	V1_meta=meta_total[meta_total.get('ecephys_structure_acronym')=='VISp'].get(['unit_id','ecephys_session_id','ecephys_probe_id','specimen_id']).drop_duplicates()

	#Convert DataFrame to csv
	V1_meta.to_csv('V1_meta',index=False)

# Can be ran as script
if __name__ == "__main__":
	main()