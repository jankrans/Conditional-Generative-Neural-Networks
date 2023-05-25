import numpy as np
import pandas as pd 

"""
    Every one of these functions expects a full interval df (so zero and NaN intervals) 
    but will only give a predictions for the zero intervals 
"""

def sign_change_intervals(interval_df):
    zero_intervals = interval_df.query('interval_value == 0')
    short_zero_intervals = (
        zero_intervals
        .replace({'start':np.nan, 'end': np.nan})
        .dropna(subset = ['0th_value_after_end', 'value_before_start'])
        .query('interval_length == 1')
    )
    sign_change_intervals = np.sign(short_zero_intervals['value_before_start']) == - np.sign(short_zero_intervals['0th_value_after_end'])
    result = pd.Series(index = zero_intervals.index, dtype = 'object')
    
    # a short zero interval with a sign change is normal
    result.loc[sign_change_intervals[sign_change_intervals].index] = False
    
    return result
    
    
def low_consumption_on_both_sides_intervals(interval_df): 
    zero_intervals = interval_df.query('interval_value == 0')
    short_zero_intervals = (
        zero_intervals
        .replace({'start':np.nan, 'end': np.nan})
        .dropna(subset = ['0th_value_after_end', 'value_before_start'])
        .query('interval_length == 1')
    )
    low_consumption = (np.abs(short_zero_intervals['value_before_start']) < 0.1) & (np.abs(short_zero_intervals['0th_value_after_end']) < 0.1)
    result = pd.Series(index = zero_intervals.index, dtype = 'object')
    
    # a short zero interval with low consumption on both sides is normal
    result.loc[low_consumption[low_consumption].index] = False
    
    return result

def collective_error_intervals(interval_df, threshold = 2): 
    # count how much each start time occurs
    interval_counts = interval_df.reset_index().groupby('start')[['meterID', 'year']].size()
    # add this to the interval df as a column
    intervals_with_count = interval_df.join(interval_counts.to_frame('count'), on = ['start'])

    # only use the intervals with a very high count
    intervals_with_count = intervals_with_count[intervals_with_count['count'] >= 33] 

    # filter each group of intervals that start on the same moment, only allow intervals with the most common length +- a threshold (in this case 2)
    def filter_groups(df): 
        most_common_value = df.interval_length.value_counts().idxmax()
        return df[(df.interval_length >= most_common_value -threshold) & (df.interval_length <= most_common_value + threshold) ]
    intervals_with_count = intervals_with_count.groupby('start_time').apply(filter_groups).droplevel(0)
    
    # each of the intervals that remains is thus a collective data problem and is a data error
    collective_data_problems  = pd.Series(index = interval_df.index, dtype = 'object')
    collective_data_problems.loc[intervals_with_count.index] = True
    return collective_data_problems
    