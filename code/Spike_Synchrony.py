# Look into text editor addons
import pickle
import os
import numpy as np
import pandas as pd
import pyspike as spk
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import neo
import elephant
import quantities as pq

#Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding'
DATA_LOC=f'{PROJECT_PATH}/data/spike_data'

def comp_cov(pop_spikes):
    isi = np.diff(pop_spikes)
    cov = np.std(isi) / np.mean(isi)   
    return cov

#s=pd.read_pickle(f"C:\\Users\\User\\visual_encoding\\data\\pickles\\valid_stationary_epochs.pkl")
#r=pd.read_pickle(f"C:\\Users\\User\\visual_encoding\\data\\pickles\\valid_running_epochs.pkl")

def spike_synchrony(BRAIN_STRUCTURE, valid_stationary_epochs, valid_running_epochs,epoch_length):
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()
	session_ids=sessions[sessions.get('session_type')=='functional_connectivity'].index

	#Choose a random epoch out of list of valid epochs and shorten epoch to appropriate length
	running_epochs={}
	stationary_epochs={}

	np.random.seed(101)

	for session_id in session_ids:
	    if valid_running_epochs[session_id]:
	        begin=valid_running_epochs[session_id][np.random.choice(len(valid_running_epochs[session_id]))]
	        running_epochs[session_id]=[begin[0],begin[0]+epoch_length]
	    else:
	        running_epochs[session_id]=[]

	    if valid_stationary_epochs[session_id]:
	        begin=valid_stationary_epochs[session_id][np.random.choice(len(valid_stationary_epochs[session_id]))]
	        stationary_epochs[session_id]=[begin[0],begin[0]+epoch_length]
	    else:
	        stationary_epochs[session_id]=[]


	#Obtain spike times within epochs, apply spike metrics, create dataframe 
	epoch_run_spike_times={}
	epoch_sta_spike_times={}

	epoch_run_bin_spikes={}
	epoch_sta_bin_spikes={}

	raw_run_spikes={}
	raw_sta_spikes={}

	covs_sta, covs_run=[],[]
	mfr_sta, mfr_run=[],[]
	spkdist_sta, spkdist_run=[],[]
	spksync_sta, spksync_run=[],[]
	cc_sta, cc_run=[],[]

	for ses_id in session_ids:
	    all_spikes=[]
	    if os.path.exists(f'{DATA_LOC}/{str(ses_id)}_{BRAIN_STRUCTURE}_spike_times.pkl'):
	        spikes=pd.read_pickle(f'{DATA_LOC}/{str(ses_id)}_{BRAIN_STRUCTURE}_spike_times.pkl')
	        s=True

	        for unit in spikes:
	            all_spikes.append(spikes[unit])

	    raw_run_spikes[ses_id]=[]
	    raw_sta_spikes[ses_id]=[]

	    epoch_run_spike_times[ses_id]=[]
	    epoch_sta_spike_times[ses_id]=[]

	    epoch_run_bin_spikes[ses_id]=[]
	    epoch_sta_bin_spikes[ses_id]=[]
	    #print(ses_id)
	    if running_epochs[ses_id] and all_spikes:
	        pop_spikes=[running_epochs[ses_id][0]]
	        fr=0
	        for i in range(len(all_spikes)):
	            unit_times=all_spikes[i]
	            raw_run_spikes[ses_id].append(unit_times[(unit_times>running_epochs[ses_id][0]) & (unit_times<running_epochs[ses_id][1])])
	            fr+=len(raw_run_spikes[ses_id][i])/epoch_length
	            pop_spikes = np.hstack([pop_spikes,raw_run_spikes[ses_id][i]])
	            epoch_run_spike_times[ses_id].append(spk.SpikeTrain(raw_run_spikes[ses_id][i], running_epochs[ses_id]))
	            neo_obj=neo.SpikeTrain(times=raw_run_spikes[ses_id][i], units='sec', t_start=running_epochs[ses_id][0], t_stop=running_epochs[ses_id][1])
	            epoch_run_bin_spikes[ses_id].append(neo_obj)
	        mfr_run.append(fr/len(raw_run_spikes[ses_id]))
	        pop_spikes = np.sort(pop_spikes)
	        covs_run.append(comp_cov(pop_spikes))
	        spkdist_run.append(spk.spike_distance(epoch_run_spike_times[ses_id]))
	        spksync_run.append(spk.spike_sync(epoch_run_spike_times[ses_id]))
	        cc_run.append(elephant.spike_train_correlation.correlation_coefficient(elephant.conversion.BinnedSpikeTrain(epoch_run_bin_spikes[ses_id], bin_size=1 * pq.s)))
	    else:
	        covs_run.append(None)
	        mfr_run.append(None)
	        spkdist_run.append(None)
	        spksync_run.append(None)
	        cc_run.append(None)

	    if stationary_epochs[ses_id] and all_spikes:
	        pop_spikes=[stationary_epochs[ses_id][0]]
	        fr=0
	        for i in range(len(all_spikes)):
	            unit_times=all_spikes[i]
	            raw_sta_spikes[ses_id].append(unit_times[(unit_times>stationary_epochs[ses_id][0]) & (unit_times<stationary_epochs[ses_id][1])])
	            fr+=len(raw_sta_spikes[ses_id][i])/epoch_length
	            pop_spikes = np.hstack([pop_spikes,raw_sta_spikes[ses_id][i]])
	            epoch_sta_spike_times[ses_id].append(spk.SpikeTrain(raw_sta_spikes[ses_id][i], stationary_epochs[ses_id]))
	            neo_obj=neo.SpikeTrain(times=raw_sta_spikes[ses_id][i], units='sec', t_start=stationary_epochs[ses_id][0], t_stop=stationary_epochs[ses_id][1])
	            epoch_sta_bin_spikes[ses_id].append(neo_obj)
	        mfr_sta.append(fr/len(raw_sta_spikes[ses_id]))
	        pop_spikes = np.sort(pop_spikes)
	        covs_sta.append(comp_cov(pop_spikes))
	        spkdist_sta.append(spk.spike_distance(epoch_sta_spike_times[ses_id]))
	        spksync_sta.append(spk.spike_sync(epoch_sta_spike_times[ses_id]))
	        cc_sta.append(elephant.spike_train_correlation.correlation_coefficient(elephant.conversion.BinnedSpikeTrain(epoch_sta_bin_spikes[ses_id], bin_size=1 * pq.s)))
	        #Add spike sync metric, elephant correlation coeff
	    else:
	        covs_sta.append(None)
	        mfr_sta.append(None)
	        spkdist_sta.append(None)
	        spksync_sta.append(None)
	        cc_sta.append(None)
	    #print('\n')

	#Form lists/arrays to construct final DataFrame
	mfr=mfr_sta+mfr_run
	covs=covs_sta+covs_run
	spkdist=spkdist_sta+spkdist_run
	spksync=spksync_sta+spksync_run
	cc=cc_sta+cc_run

	ses_list=2*list(session_ids)
	struc=len(mfr)*[BRAIN_STRUCTURE]
	eps=[stationary_epochs[ses_id] for ses_id in session_ids]+[running_epochs[ses_id] for ses_id in session_ids]
	states=len(session_ids)*['stationary']+len(session_ids)*['running']

	synchrony_df=pd.DataFrame({'sessions':ses_list,'mean_firing_rate':mfr,'cov':covs,'SPIKE_distance':spkdist, 'SPIKE_sync': spksync, 'corr_coeff':cc,'epoch': eps,'brain_structure':struc,'state':states})

	return synchrony_df