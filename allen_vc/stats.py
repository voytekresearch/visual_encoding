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
    regions  = df.get('brain_structure').unique()
    
    for region in regions:
        print(f'REGION: {region}\n\n' + 'x'*100 + '\n')
        rdf = df[df['brain_structure']==region]
        for metric in metrics:
            print(f'METRIC: {metric}\n')
                
            s_data = rdf[rdf.get(condition)==states[0]]
            s = s_data.get(metric).dropna()
            print(f'Behavior: {states[0]}\nN = {len(s)}\nMean = {np.mean(s)}\nStdev = {np.std(s)}\n')
                
            r_data = rdf[rdf.get(condition)==states[-1]]
            r = r_data.get(metric).dropna()
            print(f'Behavior: {states[-1]}\nN = {len(r)}\nMean = {np.mean(r)}\nStdev = {np.std(r)}\n')

            i = sts.ttest_ind(s, r)
            print(f'Independent T-Test (All data)\n{i}\n')
            
            if paired_ttest:
                valid_sessions = rdf[np.array([False if len(rdf[rdf.get('session')==ses_id])!=2 \
                    else True for ses_id in rdf.get('session')])]
                s = valid_sessions[valid_sessions.get('behavior')==states[0]].get(metric)
                r = valid_sessions[valid_sessions.get('behavior')==states[1]].get(metric)
                p = sts.ttest_rel(s, r)
                print(f'Paired T-Test\n{p}\n')

            print('\n' + '-'*100 + '\n')


def create_r_matrix(statistics, mat):    
    """
    Create a matrix of Pearson correlation coefficients between spike statistics and PSD data.

    Parameters
    ----------
    statistics : array_like
        Array of spike statistics.
    mat : array_like
        3D array of PSD data.

    Returns
    -------
    r_mat : array_like
        2D array of Pearson correlation coefficients.
    """
    import scipy.stats as sts

    # initialize matrix for storage of r values
    r_mat = []
    p_mat = []
    
    # loop through data in mat
    for i, row in enumerate(mat):
        r_mat_row = []
        p_mat_row = []
        for j, col in enumerate(row):
            all_trials = mat[i,j,:]
            
            # disregard invalid ranges
            if all(np.isnan(all_trials)):
                results = [np.nan, np.nan]
            else:
                # some trials have NaN PSD data so filter those out
                results = sts.pearsonr(statistics[~np.isnan(all_trials)], \
                                      all_trials[~np.isnan(all_trials)])
            r_mat_row.append(results[0])
            p_mat_row.append(results[1])
        r_mat.append(r_mat_row)
        p_mat.append(p_mat_row)
        
    return np.array(r_mat), np.array(p_mat)
