"""
Retrieve spiking data for all sessions with given BRAIN_STRUCTURES.
"""

# imports
import os
import pandas as pd
import pickle

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand

# settings - directories
MANIFEST_PATH = "E:/datasets/allen_vc"
PROJECT_PATH = "G:/Shared drives/visual_encoding"
REPO_PATH = 'C:/Users/User/visual_encoding' # code repo r'C:\Users\micha\visual_encoding
RELATIVE_PATH_OUT = "data/spike_data/spike_times"

# settings - dataset details
BRAIN_STRUCTURES = ['VISp','LGd']


def main():

    # Define/create directories
    for base_path in [PROJECT_PATH, MANIFEST_PATH]:
        dir_results = f'{base_path}/{RELATIVE_PATH_OUT}'
        if not os.path.exists(dir_results): 
            os.makedirs(dir_results)
    
    # Create Allensdk cache object
    manifest_path = f"{MANIFEST_PATH}/manifest_files/manifest.json"

    for structure in BRAIN_STRUCTURES:
        print(f'Analyzing Brain Structure: {structure}')
        metadata=pd.read_csv(f'{PROJECT_PATH}\\data\\brain_structure_DataFrames\\{structure}_DataFrame.csv')\
            .set_index('id')

        # loop through sessions
        for session_id in metadata.get('ecephys_session_id').unique():
            print(f'Analyzing session: \t{session_id}')
            
            # Get spiking data
            spiketrains = gen_neo_spiketrains(session_id, manifest_path, metadata, structure)

            # save to file
            for base_path in [PROJECT_PATH, MANIFEST_PATH]:
                fname_out = f"{base_path}/{RELATIVE_PATH_OUT}/{str(session_id)}_{structure}.pkl"
                h = open(fname_out,"wb")
                pickle.dump(spiketrains, h)
                h.close()


def gen_neo_spiketrains(session_id, manifest_path, metadata, brain_structure=None):
    """
    load spiking data for a session and reformat as Neo object.
    Parameters
    ----------
    session : AllenSDK session object
        AllenSDK session object.
    manifest_path : str
        path to AllenSDK manifest file.
    metadata : Pandas DataFrame
        contains information for unit annotations.
    brain_structure : str, optional
        include to filter results by brain structure. The default is None.
    Returns
    -------
    spiketrains : Neo SpikeTrains object
        Neo SpikeTrains object
    """

    # imports
    import neo
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    # Create Allensdk cache object
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    #Get all session info
    session = cache.get_session_data(session_id)

    # Create Neo SpikeTrain object
    spiketrains = []

    # Retrive raw spikes and save as a group containing a single Neo SpikeTrain
    if brain_structure:
        for unit in session.units[session.units.get('ecephys_structure_acronym')==brain_structure].index:
            annotations = dict(metadata.loc[unit])
            annotations['unit_id'], annotations['region'] = unit, brain_structure
            session_spikes = session.spike_times[unit]
            spiketrains.append(neo.SpikeTrain(times=session_spikes, \
                units='sec', t_stop=session_spikes[-1], **annotations))
    else:
        for unit in session.units.index:
            annotations = dict(metadata.loc[unit])
            annotations['unit_id'] = unit
            session_spikes = session.spike_times[unit]
            spiketrains.append(neo.SpikeTrain(times=session_spikes, \
                units='sec', t_stop=session_spikes[-1], **annotations))

    return spiketrains


if __name__ == '__main__':
    main()