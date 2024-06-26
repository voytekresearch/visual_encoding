"""
THis module contains general/misc utility functions.

Functions:
----------
hour_min_sec : Convert seconds to hours, minutes, and seconds.
print_time_elapsed : Print time elapsed since start time.
save_pkl : Save dictionary as pickle.
params_to_df : Convert peak_params object to dataframe.
combine_spike_lfp_dfs : Combine spike and LFP dataframes.
channel_medians : Take median across channels for param data.
ellipse_area : Calculate the area of an ellipse.
knee_freq : Convert specparam knee parameter to Hz.

"""


# imports
import numpy as np
import pandas as pd


def hour_min_sec(duration):
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs

def print_time_elapsed(start):
    import time
    duration = time.time() - start
    hours, mins, secs = hour_min_sec(duration)
    print(f"{hours} hours, {mins} minutes, and {secs :0.1f} seconds")


def save_pkl(dictionary, fname_out, method='pickle'):
    """
    save dictionary as pickle

    Parameters
    ----------
    dictionary : dict
        dictionary to be saved to file.
    fname_out : str
        filename for output.
    method : str, optional
        what method (pickle or joblib) to use to saving. 
        OS users were having issues saving large files with pickle.
        The default is 'pickle'.

    Returns
    -------
    None.

    """
    # imports
    import pickle

    # save pickle
    f = open(fname_out, "wb")
    pickle.dump(dictionary, f)
    f.close()


def params_to_df(params, max_peaks):
    # get per params
    df_per = pd.DataFrame(params.get_params('peak'),
        columns=['cf','pw','bw','idx'])

    # get ap parmas
    df_ap = pd.DataFrame(params.get_params('aperiodic'),  
        columns=['offset', 'knee', 'exponent'])

    # get quality metrics
    df_ap['r_squared'] = params.get_params('r_squared')

    # initiate combined df
    df = df_ap.copy()
    columns = []
    for ii in range(max_peaks):
        columns.append([f'cf_{ii}',f'pw_{ii}',f'bw_{ii}'])
    df_init = pd.DataFrame(columns=np.ravel(columns))
    df = df.join(df_init)

    # app peak params for each peak fouond
    for i_row in range(len(df)):
        # check if row had peaks
        if df.index[i_row] in df_per['idx']:
            # get peak info for row
            df_ii = df_per.loc[df_per['idx']==i_row].reset_index()
            # loop through peaks
            for i_peak in range(len(df_ii)):
                # add peak info to df
                for var_str in ['cf','pw','bw']:
                    df.at[i_row, f'{var_str}_{i_peak}'] = df_ii.at[i_peak, var_str]
    
    return df
    

def combine_spike_lfp_dfs(spike_df, lfp_df, ses_id, region, state=None):
    '''
    Combines spike and LFP dataframes given session ID and region.

    Parameters
    ----------
    spike_df : dataframe
        Dataframe containing spike parameters.
    lfp_df : dataframe
        Dataframe containing LFP parameters.
    ses_id : int
        Session ID for dataframes.
    region : str
        Brain region for dataframes.
    state : str, optional
        Running state for dataframes.

    Returns
    -------
    df : dataframe
        Combined dataframe of spike and LFP parameters.
    '''

    # filter region
    reg_spike_df = spike_df[spike_df.get("brain_structure")==region]

    # filter session
    ses_spike_df = reg_spike_df[reg_spike_df.get("session")==ses_id]
    ses_lfp_df = lfp_df[lfp_df.get("session")==ses_id]
    
    # take median across channels for param data
    chan_med_lfp_df = channel_medians(ses_lfp_df, ses_lfp_df.columns).drop(columns="chan_idx")
    
    if state is not None:
        state_spike_df = ses_spike_df[ses_spike_df.get("running")==state]
        return chan_med_lfp_df.merge(state_spike_df)
    
    return chan_med_lfp_df.merge(ses_spike_df)


def channel_medians(lfp_df, col_names):
    """
    This function takes in a dataframe of LFP data and a list of column names and returns a dataframe of the medians for each channel for each epoch.

    Parameters
    ----------
    lfp_df : dataframe
        A dataframe of LFP data
    col_names : list
        A list of column names

    Returns
    -------
    dataframe
        A dataframe of the medians for each channel for each epoch
    """
    
    #imports
    import pandas as pd
    
    medians = [0]*len(col_names)
    for epoch in lfp_df.get("epoch_idx").unique():
        epoch_df = lfp_df[lfp_df.get("epoch_idx")==epoch]
        medians = np.vstack((medians, epoch_df.median()))

    medians = np.delete(medians, (0), axis=0)
    return pd.DataFrame(data = medians, columns = col_names)#.drop(columns = 'chan_idx')


def ellipse_area(a, b):
    """
    This function takes in the radii of an an ellipse in arbitrary order and returns the ellipse's area.

    Parameters
    ----------
    a : float/int
        radius 1
    b : float/int
        radius 2

    Returns
    -------
    float
        Area of the ellipse
    """
    return np.pi*a*b


def knee_freq(knee, exponent):
    """
    Convert specparam knee parameter to Hz.

    Parameters
    ----------
    knee, exponent : 1D array
        Knee and exponent parameters from specparam.

    Returns
    -------
    knee_hz : 1D array
        Knee in Hz.
    """
    
    # check if input is float or array
    if isinstance(knee, float):
        knee_hz = knee**(1/exponent)

    else:
        knee_hz = np.zeros_like(knee)
        for ii in range(len(knee)):
            knee_hz[ii] = knee[ii]**(1/exponent[ii])
        
    return knee_hz