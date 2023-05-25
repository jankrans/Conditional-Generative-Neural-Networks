import numpy as np
import pandas as pd
from tqdm import tqdm
from energyclustering.data.preprocessing.interval_information import get_interval_df

def replace_data_problems_with_NaN(data_df, interval_df, is_error):
    interval_with_error = interval_df.join(is_error.to_frame('is_error'))
    data_df_with_errors = data_df.copy()
    for index, row in tqdm(interval_with_error[interval_with_error.is_error == True].iterrows()):
        meterID, year, start, end = index
        data_df_with_errors.loc[(meterID, year), :].iloc[start:end+1] = np.NAN
    return data_df_with_errors

def get_cumulative_measurements_simple(data_df, info_df, interval_df, iqr_multiplier = 1.5):
    """
        A simple version of the cumulative measurement detection based on Kostas idea to use simple boxplot techniques and the connection capacity
        The more complex one is implemented under real_data_exploration/handling_zeros_and_nans

    """

    is_error = pd.Series([False] * len(interval_df), index=interval_df.index)

    print(
        f'boxplot outlier detection: everything outside Q1 - {iqr_multiplier:.2f}, Q3 + {iqr_multiplier:0.2f} is an error',
        end="")
    iqr_outliers = get_boxplot_peaks(interval_df, data_df, iqr_multiplier=iqr_multiplier, iqr_lower_bound = 0.25)
    iqr_outlier_profiles = iqr_outliers[iqr_outliers].index
    is_error.loc[iqr_outlier_profiles] = True
    print("DONE")

    print("If it is not a single value peak correct it...", end = '')
    both_values_known = interval_df[(interval_df['0th_value_after_end'] != 'end') & (interval_df['1th_value_after_end'] != 'end')]
    is_not_single_value_peak = both_values_known['0th_value_after_end'].astype('float') < both_values_known['1th_value_after_end'].astype('float')
    profiles = is_not_single_value_peak[is_not_single_value_peak].index
    is_error.loc[profiles] = False
    print("DONE")


    print(
        "connection_and_pv_power_peaks: everything higher than connection power or lower than minus PV-power is an error",
        end="")
    connection_power_peaks = get_connection_and_pv_power_peaks(interval_df)
    is_error[connection_power_peaks] = True
    print("DONE")
 
    return is_error

def replace_connection_and_pv_power_peaks_with_nan(data_df, info_df):
    PV_power = info_df['PV_power'].astype('float')
    connection_power = info_df['connection_power'].astype('float')
    # gt is greater than, lt is less than
    clearly_wrong_peak_indicator = data_df.gt(connection_power, axis = 0) | data_df.lt(-connection_power, axis = 0) | data_df.lt(- PV_power, axis = 0)
    new_data_df = data_df.copy()
    new_data_df[clearly_wrong_peak_indicator] = np.NAN
    return new_data_df

def get_connection_and_pv_power_peaks(interval_df):
    """
        returns a boolean series that for each interval indicates whether or not is followed by a cumulative value

        The cumulative value detection is based on the connection_power and PV_power of each meter
    """
    peak_value = interval_df['0th_value_after_end'].replace({'end': np.nan}).astype('float')
    connection_power = interval_df.connection_power.astype('float')
    PV_power = interval_df.PV_power.astype('float')
    # x < NaN is false so this last part of the equations evaluates to false if PV_power is NaN
    cumulative = (peak_value > connection_power) | (peak_value < - connection_power) | (peak_value < -PV_power)
    return cumulative

def get_boxplot_peaks(interval_df, data_df, iqr_multiplier = 1.5, iqr_lower_bound = 0.25):
    # first find the bounds to use
    def get_bounds(row):
        q1 = row.quantile(0.25)
        q3 = row.quantile(0.75)
        # lower bound to solve some issues with profiles with lots of zeros
        iqr = max(q3-q1, iqr_lower_bound)
        lower_bound = q1- iqr*iqr_multiplier
        upper_bound = q3+ iqr*iqr_multiplier
        return lower_bound, upper_bound
    profiles_with_problems = interval_df.index.droplevel([2,3]).unique()
    data_with_problems:pd.DataFrame = data_df.loc[profiles_with_problems]
    bounds = data_with_problems.apply(get_bounds, axis = 1,result_type = 'expand').set_axis(['lower_bound', 'upper_bound'], axis = 1)
    interval_df = interval_df[interval_df['0th_value_after_end']!='end'].join(bounds)
    is_peak = (interval_df['0th_value_after_end'] < interval_df['lower_bound']) | (interval_df['0th_value_after_end'] > interval_df['upper_bound'])
    return is_peak

