import pandas as pd 
import numpy as np
import dask.dataframe as dd #conda install dask
from dask.distributed import Client

def get_interval_df(data_df, info_df, keep_zero= True, keep_nan = True):
    """
        Returns a dataframe with index profile, year, start, end, interval_value, interval_length, start_time, end_time
        and columns: 
            - interval_length
            - start_time
            - end_time
            - 0th_value_after_end
            - 1th_value_after_end 
            - connection_power
            - PV_power 
    """
    interval_df = (
        data_df
        # only keep profiles with data problems (e.g. with zeros or NaNs depending on the parameters)
        .pipe(filter_profiles_with_data_problems, keep_zero, keep_nan)
        # converts the profiles to intervals 
        .pipe(get_zero_nan_intervals)
        # only keep the intervals with the desired type (zero or NaN)
        .pipe(filter_intervals_on_value, keep_zero, keep_nan)
        #
        .pipe(remove_missing_hour_dst)
        .pipe(add_values_after_interval, data_df)
        .pipe(add_value_before_interval, data_df)
        .pipe(add_connection_and_pv_power, info_df)
    )
    return interval_df
        
        
def filter_profiles_with_data_problems(data_df, keep_zero = True, keep_nan = True): 
    """
     Only keeps the profiles that have data problems 
         if keep_zero is true retain profiles with zeros
         if keep_nan is true retain profiles with nans 
    """
    assert keep_zero or keep_nan, 'at least one of keep_zero or keep_nan has to be True'
    selector = None 
    if keep_zero: 
        has_zeros = (data_df == 0).any(axis = 1)
        if selector is None: 
            selector = has_zeros 
        else: 
            selector = selector | has_zeros
    if keep_nan: 
        has_nans = data_df.isna().any(axis = 1) 
        if selector is None: 
            selector = has_nans
        else: 
            selector = selector | has_nans    
    return data_df[selector]

        
def get_zero_nan_intervals(data_df): 
    """
        returns a dataframe with index (meterID, year, start, end) and columns interval_value, start_time, end_time
        where each row represents an interval
        
        note start is inclusive, end is exclusive
        start_time is exclusive and end_time is exclusive (for plotting)
    """
    # for each profile look for the intervals
    # this has to happen on a profile to profile basis
    dfs = []
    for (meterID, year), row in data_df.iterrows(): 
        nan_df = value_interval( meterID, year,row, np.NaN)
        zero_df = value_interval( meterID, year, row, 0)
        dfs.append(nan_df)
        dfs.append(zero_df)
    full_df = pd.concat(dfs, axis = 0)
    
    # add interval_length
    full_df['interval_length'] = full_df.end - full_df.start
    
    # add start_time and end_time
    full_df['start_time'] = data_df.columns[full_df['start']] - pd.Timedelta('15min')
    # end time with a detour because the end_time might not exist in the columns! 
    full_df['end_time'] = data_df.columns[full_df['end']-1] 
    full_df['end_time'] += pd.Timedelta('15min')
    full_df = full_df.set_index(['start', 'end'], append = True)
    return full_df
        
        
def value_interval(meterID, year, a, value):
        """
            Makes a dataframe containing the start and end of each interval (only the longest intervals) that only contains value
        """
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        if np.isnan(value):
            iszero = np.concatenate(([0], np.isnan(a).view(np.int8), [0]))
        else: 
            iszero = np.concatenate(([0], np.equal(a, value).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        df = pd.DataFrame(ranges, columns = ['start', 'end'])
        df['meterID'] = meterID
        df['year'] = year
        df['interval_value'] = value
        return df.set_index(['meterID', 'year'])
        
        
def filter_intervals_on_value(interval_df, keep_zero = True, keep_nan = True): 
    assert keep_zero or keep_nan, 'at least one should be true'
    if not keep_zero: 
        interval_df = interval_df[interval_df.interval_value != 0]
    if not keep_nan: 
        interval_df = interval_df[~interval_df.interval_value.isna()]
    return interval_df

        
def remove_missing_hour_dst(interval_df): 
    # TODO fix only works for 2016! 
    interval_df = interval_df[~((interval_df.start_time == '2016-03-27 02:00:00') & (interval_df.end_time == '2016-03-27 03:00:00'))]
    interval_df = interval_df[~((interval_df.start_time == '2016-03-27 01:45:00') & (interval_df.end_time == '2016-03-27 03:00:00'))]
    return interval_df 


def add_values_after_interval(interval_df, data_df): 
    def values_after_end(row):
        meterID, year, start, end = row.name
        # if end is to large
        try:
            first_value = data_df.at[(meterID,year), data_df.columns[end]]
        except: 
            first_value = 'end'
        try:
            second_value = data_df.at[(meterID,year), data_df.columns[end+1]]
        except: 
            second_value = 'end'
        return first_value, second_value
    after_values_df = interval_df.apply(values_after_end, axis = 1, result_type = 'expand').rename(columns = lambda x: f'{x}th_value_after_end')
    interval_df = pd.concat([interval_df, after_values_df], axis = 1)
    return interval_df

def add_value_before_interval(interval_df, data_df): 
    def values_after_end(row):
        meterID, year, start, end = row.name
        # if end is to large
        try:
            first_value = data_df.at[(meterID,year), data_df.columns[start-1]]
        except: 
            first_value = 'start'
        
        return first_value
    after_values_df = interval_df.apply(values_after_end, axis = 1, result_type = 'expand').to_frame('value_before_start')
    interval_df = pd.concat([interval_df, after_values_df], axis = 1)
    return interval_df

def add_connection_and_pv_power(interval_df, info_df): 
    connection_power = info_df[['connection_power', 'PV_power']].astype('float')
    interval_df = interval_df.join(connection_power / 4)
    return interval_df


