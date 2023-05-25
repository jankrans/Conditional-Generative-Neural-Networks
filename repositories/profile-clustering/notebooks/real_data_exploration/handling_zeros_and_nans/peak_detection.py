import numpy as np
from math import ceil
import pandas as pd
from statistical_models import KDEDistribution
from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client


"""
    GENERAL NOTE: slicing with datetime includes the endpoint!!!! 
"""
def get_cumulative_value_detections(data_df, interval_df, model = None, context_size = '6H', reference_day_window = 50, k = 5, return_all_info = False, n_threads = 1, result_dir=None):
    # If you pass a dask client this client will be used to speed up the process
    if model is None: 
        model = lambda: KDEDistribution(0.99, 0.07)
    # all the individual detection methods 
    print('connection power peaks...', end = "")
    if result_dir is not None and (result_dir/'power_peaks.pkl').exists(): 
        connection_power_peaks = None
    else: 
        connection_power_peaks = get_connection_and_pv_power_peaks(interval_df)
        if result_dir is not None: 
            connection_power_peaks.to_pickle(result_dir/'power_peaks.pkl')
    print('DONE')
    print('global peaks...', end = "")
    if result_dir is not None and (result_dir/'global_peaks.pkl').exists(): 
        global_kde_peaks = None
    else: 
        global_kde_peaks = get_model_based_global_peaks(data_df, interval_df, model, return_models = False)
        if result_dir is not None: 
            global_kde_peaks.to_pickle(result_dir/'global_peaks.pkl')
    print("DONE")
    print('low start and low end...', end = '')
    if result_dir is not None and (result_dir/'low_before_and_after.pkl').exists(): 
        low_start_and_low_end = None
    else: 
        low_start_and_low_end = get_before_and_after_lows(interval_df)
        if result_dir is not None: 
            low_start_and_low_end.to_pickle(result_dir/'low_before_and_after.pkl')
    print('DONE')
    print('knn similarity based peaks...', end = '')
    same_day_intervals = interval_df[interval_df.start_time.dt.date == interval_df.end_time.dt.date]
    similarity_peaks = get_knn_similarity_based_peaks(data_df, same_day_intervals, context_size, reference_day_window, k, n_threads = n_threads)
    print('DONE')
    
    # combine everything to make a single prediction 
    final_predictions = pd.Series(index = interval_df.index)
    
    if connection_power_peaks is None: 
        connection_power_peaks = pd.read_pickle(result_dir/'power_peaks.pkl')
    if global_kde_peaks is None: 
        global_kde_peaks = pd.read_pickle(result_dir/'global_peaks.pkl')
    if low_start_and_low_end is None: 
        low_start_and_low_end = pd.read_pickle(result_dir/'low_before_and_after.pkl')
        
    final_predictions[connection_power_peaks] = True
    final_predictions[final_predictions.isna()] = similarity_peaks[final_predictions.isna()]
    final_predictions[final_predictions.isna() & global_kde_peaks] = True
    final_predictions[final_predictions.isna() & low_start_and_low_end] = False
    
    if return_all_info: 
        all_info = (
        interval_df
            .drop(columns = ['0th_value_after_end', '1th_value_after_end', 'value_before_start', 'PV_power'])
            .join(connection_power_peaks.to_frame('connection_peak'))
            .join(global_kde_peaks.to_frame('kde_peak'))
            .join(similarity_peaks.to_frame('similarity_peak'))
            .join(low_start_and_low_end.to_frame('low_start_and_end'))
            .fillna({'similarity_peak': np.nan})
        )
        return final_predictions, all_info
        
    return final_predictions
        
def get_connection_and_pv_power_peaks(interval_df):
    """
        returns a boolean series that for each interval indicates whether or not is followed by a cumulative value
        
        The cumulative value detection is based on the connection_power and PV_power of each meter
    """
    peak_value = interval_df['0th_value_after_end'].replace({'end':np.nan}).astype('float')
    connection_power = interval_df.connection_power.astype('float')
    PV_power = interval_df.PV_power.astype('float')
    # x < NaN is false so this last part of the equations evaluates to false if PV_power is Na
    cumulative = (peak_value > connection_power) | (peak_value < - connection_power) | (peak_value < -PV_power)
    return cumulative

def get_model_based_global_peaks(data_df, interval_df, model, return_models = False): 
    # function to learn a model from a row
    def get_learned_model(row, model): 
        meterID, year = row.name
        interval_endings = interval_df.loc[(meterID, year), 'end_time']
        row_normal_values = row.drop(interval_endings, errors = 'ignore').dropna()
        # make a new model and train it
        fitted_model = model()
        fitted_model.old_fit(row_normal_values.to_numpy().T)
        return fitted_model
    # because not all profiles have an interval of interest!
    profiles_of_interest = interval_df.index.droplevel([2,3]).unique()
    data_subset = data_df.loc[profiles_of_interest]
    models = data_subset.apply(get_learned_model, model = model, axis = 1)
    models = interval_df[interval_df['0th_value_after_end'] != 'end'].join(models.to_frame('model'))
    is_global_peak = models.apply(lambda row: row['model'].test_value(float(row['0th_value_after_end'])), axis = 1)
    if return_models: 
        return is_global_peak, models
    return is_global_peak


def get_similarity_based_peaks(data_df, interval_df, reference_day_window = 30, context_size = '5H', return_all_info = False): 
    def match_interval_wrapper(row, data_df, reference_day_window, context_size): 
        """ 
            Simple helper function to call match_interval from an apply call of pandas 
        """
        meterID, year, _, _ = row.name 
        start_time, end_time = row['start_time'] , row['end_time']
        return match_interval(meterID, year, start_time, end_time, data_df, reference_day_window, context_size).squeeze()
    all_info = interval_df.apply(match_interval_wrapper, data_df = data_df, reference_day_window = reference_day_window, context_size = context_size, axis = 1)
    detections = all_info['best_match'] =='cumulative'
    if return_all_info: 
        return detections, all_info
    return detections

def get_knn_similarity_based_peaks(data_df, interval_df, reference_day_window= 50, context_size = '5H', k = 5, n_threads = 1): 
    def match_interval_wrapper_parallel(row, data_df, reference_day_window, context_size, k): 
        try:
            results = match_knn_then_assumption_parallel(row, data_df, reference_day_window, context_size, k)
            decision = results[0]
        except Exception as e:
            return -1
        
        if decision is None: 
            return 0
        if decision == 'cumulative':
            return 1
        return 2
    def match_interval_wrapper(row): 
        try:
            results = match_knn_then_assumption(row, data_df, reference_day_window, context_size, k)
            decision = results[0]
        except Exception as e:
            decision = None
        if decision is None: 
            return decision
        return decision == 'cumulative'
    if n_threads == 1: 
        return interval_df.apply(match_interval_wrapper, axis = 1)
    else: 
        with Client(n_workers = n_threads, local_directory = '/cw/dtailocal/jonass/') as client: # local client with 4 processes
            npartitions = len(client.scheduler_info()['workers'])*n_threads * 20
            intervals_dask = dd.from_pandas(interval_df.reset_index(), npartitions)
            dask_result = intervals_dask.apply(match_interval_wrapper_parallel, data_df = data_df, reference_day_window = reference_day_window, context_size = context_size, k = k, axis = 1, meta = 'int8')
            # wait till this is computed
            result = dask_result.compute()
        # get the result in the correct format
        result.index = interval_df.index[result.index]
        result = result.replace({-1: np.nan, 0: np.nan, 1: True, 2: False})
        return result
        

def get_before_and_after_lows(interval_df): 
    low_start_and_low_end = pd.Series(index = interval_df.index, dtype = 'bool')
    intervals_w_problems = interval_df[['0th_value_after_end', '1th_value_after_end', 'value_before_start']].isin(['start', 'end']).any(axis = 1)
    low_start_and_low_end.loc[~intervals_w_problems] = (interval_df.loc[~intervals_w_problems, ['0th_value_after_end', '1th_value_after_end', 'value_before_start']].abs() < 0.1).all(axis = 1)
    low_start_and_low_end[intervals_w_problems] = False
    return low_start_and_low_end

def match_knn_then_assumption_parallel(row, data_df, reference_day_window = 50, context_size = '4H', k = 5):
    meterID, year = row.meterID, row.year 
    # start and end time of the interval INCLUSIVE
    # so start_time is the first NaN value and end_time is the last NaN value
    start_time, end_time = row['start_time']+pd.Timedelta('15min') , row['end_time']-pd.Timedelta('15min')
    # later all timestamps will be put on the same date so do this here as well 
    start_time2, end_time2 = start_time.replace(year = 2016, month = 1, day = 1), end_time.replace(year = 2016, month = 1, day =1)
   

    # make the dataframe with all the relevant data
    search_intervals_df = construct_search_intervals(start_time, end_time, reference_day_window, context_size, data_df)
    rel_data_df = add_data_to_search_intervals(meterID, year, search_intervals_df, data_df)
    
    # handle zeros as NaN for now 
    rel_data_df = rel_data_df.replace({0, np.nan})

    # seperate the missing day from all the other days
    missing_day = rel_data_df.loc[start_time - pd.Timedelta(context_size)/2]
    reference_days = rel_data_df.drop(index = start_time-pd.Timedelta(context_size)/2)
    
    # stats on the missing day 
    min_value_missing_day, max_value_missing_day  = abs(missing_day.squeeze().min()), abs(missing_day.squeeze().max())
    max_distance = max(min_value_missing_day, max_value_missing_day) / 2 
    
    # drop reference days with data problems
    # TODO fix for zero days then this is not really correct :) 
    reference_days.dropna(inplace = True)

    # calculate the distances between the missing day and the reference days 
    distances_known_data = reference_days.apply(sim_known_data, axis = 1, missing_day = missing_day.squeeze().to_numpy(), raw = True)
    
    # sort the distances from small to large
    sorted_distances = distances_known_data.sort_values()
       
    # take the best k matches
    best_matches = reference_days.loc[sorted_distances.iloc[:k].index]
    
    # for these matches calculate how well the cumulative and real value assumption fit 
    best_match_info = pd.DataFrame(index = best_matches.index)
    peak_time = end_time2 + pd.Timedelta('15min')
    # calculate the expected value after the interval using each assumption
    best_match_info['cumulative'] = best_matches.apply(lambda x: np.sum(x.loc[start_time2: peak_time]), axis = 1)
    best_match_info['real'] = best_matches[peak_time]
    
    # calculate the difference between the observed value and the expected value
    observed_value =  missing_day.squeeze()[peak_time]
    best_match_info['observed'] = observed_value
    best_match_info['cumulative_distance'] = np.abs(best_match_info['cumulative'] - observed_value)
    best_match_info['real_distance'] = np.abs(best_match_info['real'] - observed_value)
    
    # let each profile vote
    # if distances are the same choose real measurement
    real_votes = best_match_info.real_distance <= best_match_info.cumulative_distance
    cumulative_votes = best_match_info.cumulative_distance < best_match_info.real_distance
    dont_know_votes = best_match_info[['cumulative_distance','real_distance']].min(axis = 1) > max_distance
    best_match_info.loc[real_votes, 'vote']  = 'real'
    best_match_info.loc[cumulative_votes, 'vote'] = 'cumulative'
    best_match_info.loc[dont_know_votes, 'vote'] = 'dont_know'
    
    # count votes 
    votes = best_match_info[best_match_info.vote != 'dont_know']
    vote_count = votes.vote.value_counts()
    relative_vote_count = vote_count/ len(votes)
    
    decision_certainty = relative_vote_count.max()
    if decision_certainty >= 0.80: 
        decision = relative_vote_count.idxmax()
        if decision == 'dont_know': 
            decision = None
    else: 
        decision = None
    
#     best_match_info = best_match_info[['real_distance', 'cumulative_distance']]
    
    return decision, relative_vote_count, missing_day, best_matches, best_match_info

def match_knn_then_assumption(row, data_df, reference_day_window = 50, context_size = '4H', k = 5):
    meterID, year, _, _ = row.name
    # start and end time of the interval INCLUSIVE
    # so start_time is the first NaN value and end_time is the last NaN value
    start_time, end_time = row['start_time']+pd.Timedelta('15min') , row['end_time']-pd.Timedelta('15min')
    # later all timestamps will be put on the same date so do this here as well 
    start_time2, end_time2 = start_time.replace(year = 2016, month = 1, day = 1), end_time.replace(year = 2016, month = 1, day =1)
   

    # make the dataframe with all the relevant data
    search_intervals_df = construct_search_intervals(start_time, end_time, reference_day_window, context_size, data_df)
    rel_data_df = add_data_to_search_intervals(meterID, year, search_intervals_df, data_df)
    
    # handle zeros as NaN for now 
    rel_data_df = rel_data_df.replace({0, np.nan})

    # seperate the missing day from all the other days
    missing_day = rel_data_df.loc[start_time - pd.Timedelta(context_size)/2]
    reference_days = rel_data_df.drop(index = start_time-pd.Timedelta(context_size)/2)
    
    # stats on the missing day 
    min_value_missing_day, max_value_missing_day  = abs(missing_day.squeeze().min()), abs(missing_day.squeeze().max())
    max_distance = max(min_value_missing_day, max_value_missing_day) / 2 
    
    # drop reference days with data problems
    # TODO fix for zero days then this is not really correct :) 
    reference_days.dropna(inplace = True)

    # calculate the distances between the missing day and the reference days 
    distances_known_data = reference_days.apply(sim_known_data, axis = 1, missing_day = missing_day.squeeze().to_numpy(), raw = True)
    
    # sort the distances from small to large
    sorted_distances = distances_known_data.sort_values()
       
    # take the best k matches
    best_matches = reference_days.loc[sorted_distances.iloc[:k].index]
    
    # for these matches calculate how well the cumulative and real value assumption fit 
    best_match_info = pd.DataFrame(index = best_matches.index)
    peak_time = end_time2 + pd.Timedelta('15min')
    # calculate the expected value after the interval using each assumption
    best_match_info['cumulative'] = best_matches.apply(lambda x: np.sum(x.loc[start_time2: peak_time]), axis = 1)
    best_match_info['real'] = best_matches[peak_time]
    
    # calculate the difference between the observed value and the expected value
    observed_value =  missing_day.squeeze()[peak_time]
    best_match_info['observed'] = observed_value
    best_match_info['cumulative_distance'] = np.abs(best_match_info['cumulative'] - observed_value)
    best_match_info['real_distance'] = np.abs(best_match_info['real'] - observed_value)
    
    # let each profile vote
    # if distances are the same choose real measurement
    real_votes = best_match_info.real_distance <= best_match_info.cumulative_distance
    cumulative_votes = best_match_info.cumulative_distance < best_match_info.real_distance
    dont_know_votes = best_match_info[['cumulative_distance','real_distance']].min(axis = 1) > max_distance
    best_match_info.loc[real_votes, 'vote']  = 'real'
    best_match_info.loc[cumulative_votes, 'vote'] = 'cumulative'
    best_match_info.loc[dont_know_votes, 'vote'] = 'dont_know'
    
    # count votes 
    votes = best_match_info[best_match_info.vote != 'dont_know']
    vote_count = votes.vote.value_counts()
    relative_vote_count = vote_count/ len(votes)
    
    decision_certainty = relative_vote_count.max()
    if decision_certainty >= 0.80: 
        decision = relative_vote_count.idxmax()
        if decision == 'dont_know': 
            decision = None
    else: 
        decision = None
    
#     best_match_info = best_match_info[['real_distance', 'cumulative_distance']]
    
    return decision, relative_vote_count, missing_day, best_matches, best_match_info
    
    
def match_interval(meterID, year, start_time, end_time, data_df, reference_day_window = 30, context_size = '5H'):
    """
        Function that will find the best match to the missing interval of meter meterID, year year between start_time and end_time
    """
    # make the dataframe with all the relevant data
    search_intervals_df = construct_search_intervals(start_time, end_time, reference_day_window, context_size, data_df)
    rel_data_df = add_data_to_search_intervals(meterID, year, search_intervals_df, data_df)
    
    # seperate the missing day from all the other days
    missing_day = rel_data_df.loc[start_time - pd.Timedelta(context_size)/2]
    reference_days = rel_data_df.drop(index = start_time-pd.Timedelta(context_size)/2)
    
    # drop reference days with data problems
    reference_days.dropna(inplace = True)
    
    # calculate the similarity between missing day and each reference day
    try:
        distances_real_measurement = reference_days.apply(sim_as_real_measurement, axis = 1, missing_day = missing_day.squeeze().to_numpy(), raw = True)
        distances_cum_measurement = reference_days.apply(sim_as_cumulative_measurement, axis = 1, missing_day = missing_day.squeeze().to_numpy(), raw = True)
        distances = distances_real_measurement.to_frame('real')
        distances['cumulative'] = distances_cum_measurement
    except: 
        print(f"error in profile {meterID}, {start_time}, {end_time}")
        return pd.DataFrame([[np.nan]*5], columns = ['real_distance', 'cumulative_distance', 'real_match', 'cumulative_match', 'best_match'])
    
    # calculate the smallest distances
    best_real_distance, best_cumulative_distance = distances.min(axis = 0)
    best_real_match_date = distances.index[np.argmin(distances['real'])][0] + pd.Timedelta(context_size)/2
    best_cumulative_match_date = distances.index[np.argmin(distances['cumulative'])][0] + pd.Timedelta(context_size)/2
    best_match = 'real' if best_real_distance < best_cumulative_distance else 'cumulative'
    return pd.DataFrame([[best_real_distance, best_cumulative_distance, best_real_match_date, best_cumulative_match_date, best_match]], columns = ['real_distance', 'cumulative_distance', 'real_match', 'cumulative_match', 'best_match'])


def construct_search_intervals(start_time, end_time, reference_day_window, context_size, data_df): 
    """
        This function constructs all the parts of the timeseries that we can compare with 
        (same period as the missing period with context_size /2 added to both sides for every day in 'reference_day_window' around the missing day)
    """
    # reference days
    reference_day_window_one_side = pd.Timedelta(days = ceil(reference_day_window / 2))
    reference_day_window_size = reference_day_window_one_side *2
    # context around missing interval
    context_size = pd.Timedelta(context_size)
    context_size_one_side = context_size / 2 
    # the length of each reference period
    reference_period_length = end_time - start_time + context_size
    
    # the first search interval starts at start_time of the missing interval - half the context size - half of the day window size 
    first_reference_period_start = start_time - context_size_one_side - reference_day_window_one_side
    # the final search interval starts at the start_time of the first search interval + the size of the reference window
    final_reference_period_start = first_reference_period_start + reference_day_window_size
   
    
    # the starts of all the search intervals
    search_interval_starts = pd.date_range(first_reference_period_start, final_reference_period_start, freq = 'D')
    # the ends of all the search intervals 
    search_interval_ends = pd.date_range(first_reference_period_start + reference_period_length, final_reference_period_start + reference_period_length, freq = 'D')
    
    # make a nice df with search interval information
    df = search_interval_starts.to_frame(name = 'start').reset_index(drop = True)
    df['end'] = search_interval_ends
    
    # filter out intervals that fall outside of known range
    min_date, max_date = data_df.columns.min(), data_df.columns.max()
    before_start = df['start'] < min_date 
    after_end = df['end'] > max_date
    df = df[~before_start & ~after_end]
    
    return df

def add_data_to_search_intervals(meterID,year, search_interval_df, data_df): 
    """
        Make a dataframe with the data from all the periods in search_interval_df
    """
    def get_data(row, data_df): 
        start, end = row
        return data_df.loc[(meterID,year), start:end].values 
    
    # get the actual data
    rel_data_df = search_interval_df.apply(get_data, data_df = data_df, axis = 1, result_type = 'expand')
    
    # set columns to standardised timestamps (date 2016-1-1)
    start, end = search_interval_df.iloc[0]
    new_start = start.replace(year = 2016, month = 1, day = 1)
    new_end = end.replace(year = 2016, month = 1, day = 1) +(end.date() - start.date())
    rel_data_df.columns = pd.date_range(new_start, new_end, freq = '15min')
    # set index correctly to actual timestamps 
    rel_data_df.index = pd.MultiIndex.from_frame(search_interval_df)
    return rel_data_df

def sim_known_data(full_day, missing_day): 
    # also only works for NaN days
    iszero = np.concatenate(([0], np.isnan(missing_day).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    values_to_use = np.ones(missing_day.shape).astype('bool')
    for start,end in ranges: 
        values_to_use[start:end+1] = False
    # euclidean distance of known values (without value after missing interval)
    v1 = missing_day[values_to_use]
    v2 = full_day[values_to_use]
    return np.linalg.norm(v1-v2)

def sim_as_real_measurement(full_day, missing_day): 
    iszero = np.concatenate(([0], np.isnan(missing_day).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    values_to_use = np.ones(missing_day.shape).astype('bool')
    for start,end in ranges: 
        values_to_use[start:end] = False
    # euclidean distance of known values (with value after missing interval)
    v1 = missing_day[values_to_use]
    v2 = full_day[values_to_use]
    euclidean = np.linalg.norm(v1-v2)
    
    # distances between values after missing intervals
    other_vector = []
    indices_to_use = ranges[:,1]
    other_part = np.linalg.norm(missing_day[indices_to_use]- full_day[indices_to_use])
    return euclidean + other_part

def sim_as_cumulative_measurement(full_day, missing_day): 
    iszero = np.concatenate(([0], np.isnan(missing_day).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    values_to_use = np.ones(missing_day.shape).astype('bool')
    for start,end in ranges: 
        values_to_use[start:end+1] = False
        
    # euclidean distance of known part without value after interval
    v1 = missing_day[values_to_use]
    v2 = full_day[values_to_use]
    euclidean = np.linalg.norm(v1-v2)
    
    # distance between cumulative measurements and the sum of the measurement during missing interval 
    other_vector = []
    for start, end in ranges: 
        consumption_during_missing = np.sum(full_day[start:end+1] )
        cumulative_measurement = missing_day[end]
        other_vector.append(consumption_during_missing - cumulative_measurement)
    other_part = np.linalg.norm(other_vector)
    return euclidean + other_part