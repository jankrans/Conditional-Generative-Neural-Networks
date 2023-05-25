import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_chunked
from dask.distributed import Client


def calculate_energy_score_for_daily_matrix_daily_consumption_data(probs_per_sample, daily_train_data, daily_test_data):
    """
        Calculates the energy score for each sample seperately.
        The probs_per_sample is a dataframe, each row is test sample, the columns are a multi-index of (meterID, year, date)
        Train_consumption data each row is a training year
        Test_consumption data each row is the ground truth day of the corresponding test sample

    """
    energy_scores = np.zeros((len(probs_per_sample),))
    for idx, result_series in enumerate(probs_per_sample):
        meterID, date = result_series.name

        # correct test day
        test_day = daily_test_data.loc[(meterID, date),:].to_numpy()

        # training days with non-zero probability
        train_probabilities = result_series.to_numpy()
        # missing_keys = result_series.index.difference(daily_train_data.index)
        # if len(missing_keys)> 0:
        #     print(f"missing {len(missing_keys)} from {len(result_series.index)}")
        #     print(missing_keys)
        #     raise Exception(missing_keys)
        train_days = daily_train_data.loc[result_series.index, :]

        # calculate the energy score
        energy_score = calculate_energy_score(train_probabilities, train_days, test_day)

        # return the result
        energy_scores[idx] = energy_score
    return energy_scores


def calculate_energy_score_for_daily_matrix(probs_per_sample, train_consumption_data, test_consumption_data):
    """
        Calculates the energy score for each sample seperately.
        The probs_per_sample is a dataframe, each row is test sample, the columns are a multi-index of (meterID, year, date)
        Train_consumption data each row is a training year
        Test_consumption data each row is the ground truth day of the corresponding test sample

    """
    energy_scores = np.zeros((len(probs_per_sample),))
    for idx, result_series in enumerate(probs_per_sample):
        meterID, test_date = result_series.name
        start_date = test_date.replace(year=2016)
        end_date = start_date + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        test_day = test_consumption_data.loc[(meterID,), start_date: end_date].to_numpy()

        train_probabilities = result_series.to_numpy()
        train_days = np.zeros((result_series.shape[0], test_day.shape[1]))
        for idx2, (meterID, date) in enumerate(result_series.index):
            start_date = pd.to_datetime(date).replace(year=2016)
            end_date = start_date + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            train_day = train_consumption_data.loc[(meterID,), start_date: end_date].to_numpy()
            train_days[idx2, :] = train_day

        energy_score = calculate_energy_score(train_probabilities, train_days, test_day)
        energy_scores[idx] = energy_score
    return energy_scores




def calculate_energy_score_matrix(probs_per_samples, train_consumption_data, test_consumption_data):
    """
        Calculates the energy score for each sample separately.

        This function assumes that the forecast is a discrete distribution (i.e. a finite set of possible values each with a certain probability)

        Probs per sample: a numpy array where each row contains the probabilities that different train consumption data points get sampled
        - more precisely probs_per_samples[i,j] is the probability that for test sample i, train sample j is sampled.
        - shape (nb_test_samples, nb_train_samples)

        train_consumption_data: each row contains a training sample
        test_consumption_data: each row contains a test sample
    """
    energy_scores_per_test_sample = calculate_energy_score_for_day_matrix((train_consumption_data.to_numpy(), test_consumption_data.to_numpy()), probs_per_samples)
    return pd.Series(energy_scores_per_test_sample, index = test_consumption_data.index)

def calculate_energy_score(probs_for_sample, samples, correct_profile):
    """
        samples: 2d numpy array shape (#samples, #dim) , rows are predicted samples
        probs_for_sample: 1d numpy array shape (#samples), for each sample in samples the probability that it will get sampled
        correct_profile: 1d numpy array shape (#dim), the ground truth sample
    """
    def reduce_function(chunk, start):
        sums = (chunk * probs_for_sample).sum(axis = 1)
        return sums*probs_for_sample[start:start + sums.shape[0]]

    # because the full distance array gets very very large (GB's), process the chunks seperately into the needed sum
    second_term = 0
    for chunk in pairwise_distances_chunked(samples, reduce_func = reduce_function):
        second_term += chunk.sum(axis = None)
    distances_between_test_and_training_days = euclidean_distances(samples, correct_profile.reshape((1, -1))).squeeze()
    
    first_term = np.sum(probs_for_sample * distances_between_test_and_training_days)

    return first_term - 0.5*second_term

def calculate_energy_score_df(probs_per_sample, train_consumption_data, test_consumption_data):
    # probs_per_sample is a dataframe
    # each row corresponds with a test_sample
    # each column corresponds with an index item from the train_consumption_data

    # drop training days that have all zero probabilities
    is_not_all_zero_column = (probs_per_sample > 0).any(axis = 0)
    probs_per_sample = probs_per_sample.loc[:, is_not_all_zero_column]

    all_used_training_data = train_consumption_data.loc[probs_per_sample.columns]
    distances_between_training_days = euclidean_distances(all_used_training_data)
    distances_between_training_days = pd.DataFrame(distances_between_training_days, index = probs_per_sample.columns, columns = probs_per_sample.columns)

    all_scores = np.zeros((test_consumption_data.shape[0],))

    for idx, (test_day, sample_probs) in enumerate(zip(test_consumption_data, probs_per_sample)):
        sample_probs = sample_probs[sample_probs > 0]

        prob_matrix = sample_probs*sample_probs.T
        second_term = np.sum(prob_matrix * distances_between_training_days.loc[sample_probs.columns, sample_probs.columns].values, axis = None)

        distances_between_test_and_training_days = euclidean_distances(train_consumption_data.loc[sample_probs.columns], np.array([test_day.to_numpy()])).squeeze()
        first_term = np.sum(sample_probs.values * distances_between_test_and_training_days, axis = None)

        energy_score_for_test_sample = first_term - 0.5*second_term
        # print(energy_score_for_test_sample)

        all_scores[idx] = energy_score_for_test_sample

    return pd.Series(all_scores, index = test_consumption_data.index)






def calculate_energy_score_per_day(probs_per_sample, train_consumption_data, test_consumption_data, dask_client:Client):
    """
        Old function that calculates the energy score for each day of the year separately

        (takes a long time to execute so using dask)
    """
    # all days in the data
    dates = np.unique(train_consumption_data.columns.date)

    daily_dfs = []
    for date in dates:
        start_date = date
        end_date = pd.to_datetime(start_date)+pd.Timedelta(hours = 23, minutes = 45)

        train_days = train_consumption_data.loc[:, start_date:end_date].to_numpy()
        test_days = test_consumption_data.loc[:, start_date:end_date].to_numpy()
        daily_dfs.append((train_days, test_days))

    futures = dask_client.map(calculate_energy_score_for_day_matrix, daily_dfs, probs_per_sample = probs_per_sample)
    daily_energy_scores = dask_client.gather(futures)
    return pd.DataFrame(daily_energy_scores, index = pd.to_datetime(dates)).rename_axis('test_sample_idx', axis = 1)
    # return daily_energy_scores


def calculate_energy_score_for_day_matrix(days_tuple, probs_per_sample):
    train_days, test_days = days_tuple

    # precompute distance matrix between training days (for second term energy score)
    distances_between_training_days = euclidean_distances(train_days)

    # result storage
    all_scores = np.zeros((test_days.shape[0],))

    # for each test sample
    for idx, (test_day, sample_probs) in enumerate(zip(test_days, probs_per_sample)):

        # make it 2D
        sample_probs = np.array([sample_probs])

        probabilities_matrix = sample_probs*sample_probs.T
        second_term = np.sum(probabilities_matrix*distances_between_training_days, axis = None)

        distance_between_test_and_training_days = euclidean_distances(train_days, np.array([test_day])).reshape((-1,))
        first_term = np.sum(sample_probs[0]*distance_between_test_and_training_days, axis = None)

        energy_score_for_test_sample = first_term - 0.5*second_term

        all_scores[idx] = energy_score_for_test_sample

    return all_scores






