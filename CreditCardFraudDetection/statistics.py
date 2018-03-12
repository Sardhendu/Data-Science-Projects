from __future__ import division
import numpy as np
import pandas as pd

from scipy import stats
from sklearn import preprocessing, metrics




def stats_for_continuous_col(data, continuous_cols, label_col='Class', label=None):
    '''
        data: A data Frame
        continuous_cols : List of the continuous columns for which you need the statistics
        label_col: The Label column name
        label : The label name or class for which you need the column statistics
    '''
    stats_cols = ['col_name', 'count', 'mean', 'std', 'min' ,'25%','50', '75%','max']
    col_name_arr = []
    smry_col = []
    for col_num, col_names in enumerate(continuous_cols):
        col_name_arr.append(col_names)
        if label:
            smry_stats_col = data[data[label_col]==label][col_names].describe().reset_index().set_index('index').T
        else:
            smry_stats_col = data[col_names].describe().reset_index().set_index('index').T
        smry_col.append(np.array(smry_stats_col)[0])
    col_name_arr = np.array(col_name_arr, dtype=str).reshape(-1,1)
    smry_col = np.ndarray.round(np.array(smry_col), 3)
    smry = np.column_stack((col_name_arr, smry_col))
    return  pd.DataFrame(smry, columns=stats_cols).round(2)



def stats_for_discrete_col(data, discrete_cols, label_col):
    '''
        data : A data frame
        discrete_cols : An array of columns
        label_col: A string of label column name

        Note. When two columns are similar:
                chi_pValue ~ 0
                mutual_info ~ 1
    '''
    num_instances = len(data)
    
    stats_cols = ['col_name', 'num_categories', 'chi_pValue', 'mutual_info', 'NaNs', '% NaNs']
    
    data = data[discrete_cols + [label_col]].apply(preprocessing.LabelEncoder().fit_transform)
    
    col_name_arr = []
    out_arr = []
    for col_num, col_name in enumerate(discrete_cols):
        col_name_arr.append(col_name)
        
        data[col_name] = data[col_name].astype('str')
        num_NaNs = len(data[data[col_name] == 'nan'])
        num_instances = len(data)
        num_categories = len(np.unique(np.array([data[col_name]])))
        num_labels = len(np.unique(np.array([data[label_col]])))
        
        frequency_tab = pd.crosstab(data[col_name], data[label_col], margins=True)
        observed_tab = frequency_tab.iloc[0:num_categories, 0:num_labels]
        observed_tab = pd.DataFrame(observed_tab)
        chi_stats, chi_pValue, degree_freedom, expected_tab = stats.chi2_contingency(observed=observed_tab)
        mutual_info = metrics.cluster.normalized_mutual_info_score(np.array(data[col_name]),
                                                                   np.array(data[label_col]))
        if col_num == 0:
            out_arr = np.array([num_categories, round(chi_pValue, 3), round(mutual_info, 3), num_NaNs,
                                num_NaNs / num_instances])
        else:
            out_arr = np.vstack((out_arr,
                                 np.array([num_categories, round(chi_pValue, 3), round(mutual_info, 3),
                                           num_NaNs, num_NaNs / num_instances])))
    out_arr = np.column_stack((np.array(col_name_arr).reshape(-1, 1), out_arr))
    return pd.DataFrame(out_arr, columns=stats_cols)




