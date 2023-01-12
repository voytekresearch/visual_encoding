# imports
import numpy as np
import pandas as pd

def sync_stats(df, metrics, condition, paired_ttest=False):
    """
    Computes and prints the mean, standard deviation, and t-test results for two states in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data to be analyzed.
    metrics : list
        List of metrics to be analyzed.
    condition : str
        Name of the column in the dataframe containing the states.

    Returns
    -------
    None
    """
    
    # imports
    import scipy.stats as sts
    
    states = df.get(condition).unique()
    
    for metric in metrics:
        print(f'Metric: {metric}\n')
            
        s_data = df[df.get('state')==states[0]]
        s = s_data.get(metric).dropna()
        print(f'State: {states[0]}\nN = {len(s)}\nMean = {np.mean(s)}\nStdev = {np.std(s)}\n')
            
        r_data = df[df.get('state')==states[-1]]
        r = r_data.get(metric).dropna()
        print(f'State: {states[-1]}\nN = {len(r)}\nMean = {np.mean(r)}\nStdev = {np.std(r)}\n')

        i = sts.ttest_ind(s, r)
        print(f'Independent T-Test (All data)\n{i}\n')
        
        if paired_ttest:
            valid_sessions = df[np.array([False if any(df[df.get('session_id')==ses_id]\
                .get(metric).isnull()) else True for ses_id in df.get('session_id')])]
            s = valid_sessions[valid_sessions.get('state')==states[0]].get(metric)
            r = valid_sessions[valid_sessions.get('state')==states[1]].get(metric)
            p = sts.ttest_rel(s, r)
            print(f'Paired T-Test\n{p}\n')

        print('\n\n\n')