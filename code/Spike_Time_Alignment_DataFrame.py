import os
import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#NOTE: Must run Brain_Structure_DataFrame for given brain_structure beforehand
#Final DataFrame contains ONLY units from functional_connectivity dataset

brain_structure_acronym='VISp'
stimulus_name_duration_level=['fast_pulses', 1,4]

#Function for creating DataArray 
def optotagging_spike_counts(bin_edges, trials, units, session):
    
    time_resolution = np.mean(np.diff(bin_edges))

    spike_matrix = np.zeros( (len(trials), len(bin_edges), len(units)) )

    for unit_idx, unit_id in enumerate(units.index.values):

        spike_times = session.spike_times[unit_id]

        for trial_idx, trial_start in enumerate(trials.start_time.values):

            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))

            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            spike_matrix[trial_idx, binned_times, unit_idx] = 1

    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': trials.index.values,
            'time_relative_to_stimulus_onset': bin_edges,
            'unit_id': units.index.values
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id']
    )


#Main function
def main():

	#Define data directory and create Allensdk cache object
	data_directory = 'C:\\Users\\User\\visual_encoding\\data\\manifest_files'
	manifest_path = os.path.join(data_directory, "manifest.json")
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

	units_total = cache.get_units()
	meta_total=units_total.reset_index().rename(columns={'id':'unit_id'})
	meta=meta_total[(meta_total.get('ecephys_structure_acronym')==brain_structure_acronym)&(meta_total.get('session_type')=='functional_connectivity')].get(['unit_id','ecephys_session_id','specimen_id','genotype'])

	#Creating spike_alignments for specific stimulus
	spike_alignments={}
	for session_id in np.array(meta.drop_duplicates('ecephys_session_id').get('ecephys_session_id')):
	    session=cache.get_session_data(session_id)
	    trials = session.optogenetic_stimulation_epochs[(session.optogenetic_stimulation_epochs.duration > stimulus_name_duration_level[1]-0.001) & \
                                                    (session.optogenetic_stimulation_epochs.duration < stimulus_name_duration_level[1]+0.001)&\
                                                    (session.optogenetic_stimulation_epochs.stimulus_name == stimulus_name_duration_level[0]) &\
                                                    (session.optogenetic_stimulation_epochs.level < stimulus_name_duration_level[2]+0.01)&\
                                                    (session.optogenetic_stimulation_epochs.level > stimulus_name_duration_level[2]-0.01)]
	    if trials.shape[0]==0:
	    	continue #Don't include sessions without data in parameters
	    units = session.units[session.units.get('ecephys_structure_acronym')==brain_structure_acronym]
	    time_resolution = 0.0005 # 0.5 ms bins
	    bin_edges = np.arange(-0.01, 0.025, time_resolution)
	    spike_alignments[session_id]=optotagging_spike_counts(bin_edges, trials, units, session)

	#Extract baseline and evoked rates and add to meta DataFrame
	unit_ids=np.array([])
	baseline_rates=np.array([])
	evoked_rates=np.array([])

	for session_id in spike_alignments:
	    baseline = spike_alignments[session_id].sel(time_relative_to_stimulus_onset=slice(-0.01,-0.002))
	    baseline_rate = baseline.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
	    baseline_rates= np.append(baseline_rates,np.array(baseline_rate))
	    evoked = spike_alignments[session_id].sel(time_relative_to_stimulus_onset=slice(0.001,0.009))
	    evoked_rate = evoked.sum(dim='time_relative_to_stimulus_onset').mean(dim='trial_id') / 0.008
	    evoked_rates= np.append(evoked_rates,np.array(evoked_rate))
	    unit_ids=np.append(unit_ids,np.array(baseline_rate['unit_id']))
	    

	spike_rates=pd.DataFrame({'unit_id':unit_ids,'baseline_rate':baseline_rates,'evoked_rate':evoked_rates})
	spike_rates=spike_rates.assign(unit_id=spike_rates.get('unit_id').apply(int))
	spike_rates=spike_rates.drop_duplicates()

	meta_with_spike_rates=spike_rates.merge(meta, on='unit_id')
	meta_with_spike_rates.to_csv(f'C:\\Users\\User\\visual_encoding\\data\\brain_structure_DataFrames\\{brain_structure_acronym}_with_spikes_{stimulus_name_duration_level[0]}_{str(stimulus_name_duration_level[1])}ms_{stimulus_name_duration_level[2]}level.csv',index=False)

if __name__ == '__main__':
    main()