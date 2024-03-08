"""
Load and save DataFrames including information on all units across BRAIN_STRUCTURES.
"""

# settings - dataset details
SESSION_TYPE = 'functional_connectivity'
BRAIN_STRUCTURES = ['VISp', 'LGd']

# imports
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import sys
sys.path.append("../../allen_vc")
from paths import PATH_EXTERNAL

def main():

	# identify manifest path
	manifest_path = f'{PATH_EXTERNAL}/dataset/manifest.json'

	# get unit info for each region of interest
	for structure in BRAIN_STRUCTURES:
		print(f'Getting data for: \t{structure}')
		unit_info = get_unit_info(manifest_path, brain_structure=structure, \
			session_type=SESSION_TYPE)

		# save to file (csv)
		unit_info.to_csv(f'{PATH_EXTERNAL}/data/brain_structure_DataFrames/{structure}_DataFrame.csv')


def get_unit_info(manifest_path, brain_structure=None, session_type=None):
    """
    get info about single-units, including session, subject, probe, and channel

    Parameters
    ----------
    manifest_path : str
        path to AllenSDK manifest file.
    brain_structure : str, optional
        include to filter results by brain structure. The default is None.
    session_type : TYPE, optional
        include to filter data by session type. Options include:
            'brain_observatory_1.1' and 'functional_connectivity'
            The default is None.

    Returns
    -------
    unit_info : DataFrame, optional
        unit info DataFrame

    """

    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # Get all unit info
    unit_info = cache.get_units()

    # filter by session type
    if session_type:
        unit_info = unit_info[unit_info.get('session_type')==session_type]

    # filter by brain structure
    if brain_structure:
        unit_info = unit_info[unit_info.get('ecephys_structure_acronym')==brain_structure]

    # get info of interest
    unit_info = unit_info.get(['ecephys_session_id','specimen_id',\
        'ecephys_probe_id','ecephys_channel_id'])#.drop_duplicates()

    return unit_info


# Can be ran as script
if __name__ == "__main__":
	main()