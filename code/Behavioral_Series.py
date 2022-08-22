import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#Settings
PROJECT_PATH='C:\\Users\\User\\visual_encoding'

def sig_filt_inter(time, speed, bins):
	model=interpolate.interp1d(time, speed)
	t_vals=np.arange(time[0],time[-1],bins)
	return t_vals, model(t_vals)

#Make sure epoch lengths are in order least to greatest
def get_behavioral_epochs(epoch_lengths):
	manifest_path = f"{PROJECT_PATH}/data/manifest_files/manifest.json"
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	sessions = cache.get_session_table()

	for session_id in sessions[sessions.get('session_type')=='functional_connectivity'].index:
	    #print(session_id)
	    #Extract the data using the mid_point of each timeframe to correspond with each measured velocity
	    session=cache.get_session_data(session_id)
	    run_data=session.running_speed
	    run_data=run_data.assign(mid_time=(run_data.get('start_time')+run_data.get('end_time'))/2)
	    t=np.array(run_data.get('mid_time'))
	    x=np.array(run_data.get('velocity'))
	    
	    #Create uniform set of data using interpolation
	    bins=1/2500
	    new_t, new_x=sig_filt_inter(t,x,bins)
	    
	    #Isolating the largest timeframe of spontaneous activity
	    stimuli_df=session.stimulus_presentations
	    stimuli_df=stimuli_df[stimuli_df.get('stimulus_name')=='spontaneous'].get(['start_time','stop_time'])
	    #stimuli_df=stimuli_df[stimuli_df.get('start_time')>3600]
	    stimuli_df=stimuli_df.assign(diff=stimuli_df.get('stop_time')-stimuli_df.get('start_time')).sort_values(by='diff',ascending=False)
	    start_time=stimuli_df.get('start_time').iloc[0]
	    stop_time=stimuli_df.get('stop_time').iloc[0]
	    spon_epoch=np.array([start_time,stop_time])
	    #print(f'Start Time: {start_time}, Stop Time: {stop_time}')

	    spon_time=new_t[(new_t>start_time)][new_t[(new_t>start_time)]<stop_time]
	    spon_speed=new_x[(new_t>start_time)][new_t[(new_t>start_time)]<stop_time]

	    # Applying a median filter
	    N = 501
	    y = signal.medfilt(spon_speed, [N])

	    #Save filtered data
	    np.savez(f'{PROJECT_PATH}\\data\\np files\\behavioral series\\{session_id}', time=spon_time,filtered_speed=y,spon_epoch=spon_epoch)

	return all_epoch_data