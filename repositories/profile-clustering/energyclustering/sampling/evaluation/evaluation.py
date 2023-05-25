import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pympler.asizeof import asizeof

from energyclustering.sampling.day_of_year_samplers import YearDaySampler
from energyclustering.sampling.evaluation.energy_score import \
    calculate_energy_score_for_daily_matrix_daily_consumption_data


class SamplerEvaluator:
    def __init__(self, folds, yearly_data_df, daily_data_df, yearly_info_df, daily_info_df, client, nb_tasks,
                 nb_samples=100,
                 crossval=True, use_probs=False):

        # load folds if necessary from disk
        if isinstance(folds, str) or isinstance(folds, Path):
            folds = np.load(str(folds), allow_pickle=True)

        self.folds = folds

        self.yearly_data_df = yearly_data_df
        self.daily_data_df = daily_data_df
        self.yearly_info_df = yearly_info_df
        self.daily_info_df = daily_info_df

        self.client = client
        self.nb_chunks = nb_tasks

        self.nb_samples = nb_samples
        self.crossval = crossval
        self.use_probs = use_probs

    def evaluate_and_save(self, sampler, path):
        if path.exists():
            print(f"{path.stem} already exists! Skipping...")
            return pd.read_pickle(path)
        else:
            result = self.evaluate(sampler)
            result.to_pickle(str(path))
            return result

    def evaluate(self, sampler: YearDaySampler):
        result_series = []
        total_time = 0
        for fold_idx, (training_set, testing_set) in enumerate(self.training_test_sets):
            start_time = time.time()
            # fit the sampler on the training data
            yearly_data_train, daily_data_train, yearly_info_train, daily_info_train = training_set
            sampler.fit(yearly_data_train, daily_data_train, yearly_info_train, daily_info_train)
            # sampler.clean()

            # evaluate using the testing data
            yearly_data_test, daily_data_test, yearly_info_test, daily_info_test = testing_set
            # test_data_df = test_data_df.dropna(how='any', axis=0)
            results = self.evaluate_execution_function()(sampler, daily_data_train, daily_data_test, yearly_info_test,
                                                         daily_info_test)
            result_serie = pd.Series(results, index=daily_info_test.index)
            result_series.append(result_serie)
            end_time = time.time()
            total_time += end_time - start_time
            logging.debug(
                f"Training and testing fold {fold_idx} took {pd.Timedelta(seconds=int(end_time - start_time))}")

        logging.debug(f"Training and testing all folds took {pd.Timedelta(seconds=int(total_time))}")
        return pd.concat(result_series)

    def evaluate_execution_function(self):
        if self.client is not None:
            return self.evaluate_dask__
        return self.evaluate_local_chunked__

    def evaluate_local__(self, sampler, daily_data_train, daily_data_test, yearly_info_test, daily_info_test):
        return sample_and_evaluate_local(sampler, daily_data_train, daily_data_test, yearly_info_test, daily_info_test)

    def evaluate_dask__(self, sampler, daily_data_train, daily_data_test, yearly_info_test, daily_info_test):
        nb_chunks_to_use = min(self.nb_chunks, yearly_info_test.shape[0])

        # divide the test set into multiple parts
        chunks = np.array_split(yearly_info_test.index, nb_chunks_to_use)

        # evaluate each test sample
        logging.debug(
            f'scattering training data ({asizeof(daily_data_train) / 1000000: .2f} MB) ')
        start_time = time.time()
        train_data_future = self.client.scatter(daily_data_train.copy(), broadcast=True)
        logging.debug(
            f'scattering training data took {time.time() - start_time}s')
        logging.debug(
            f'scattering sampler ({asizeof(sampler) / 1000000: .2f} MB)')
        start_time = time.time()
        sampler_future = self.client.scatter(sampler, broadcast=True)
        logging.debug(
            f'scattering sampler took {time.time() - start_time}s')
        # sampler_future = sampler
        # train_data_future = daily_data_train
        logging.debug('Setting-up all the tasks')
        tasks = [
            self.client.submit(sample_and_evaluate,
                               sampler_future,
                               train_data_future,
                               daily_data_test.loc[chunk],
                               yearly_info_test.loc[chunk],
                               daily_info_test.loc[chunk],
                               self.nb_samples,
                               self.use_probs
                               )
            for chunk in chunks
        ]
        logging.debug(f'computing the tasks')
        results = self.client.gather(tasks)
        logging.debug(f'Received results')
        return np.concatenate(results)

    def evaluate_local_chunked__(self, sampler, daily_data_train, daily_data_test, yearly_info_test, daily_info_test):
        # divide the test set into multiple parts
        chunks = np.array_split(yearly_info_test.index, self.nb_chunks)

        # evaluate each test sample
        tasks = [
            sample_and_evaluate_local(
                sampler,
                daily_data_train,
                daily_data_test.loc[chunk],
                yearly_info_test.loc[chunk],
                daily_info_test.loc[chunk],
            )
            for chunk in chunks
        ]
        results = tasks
        return np.concatenate(results)

    @property
    def training_test_sets(self):
        folds = self.folds
        assert len(folds) > 1

        if self.crossval:
            for test_idx in range(len(folds)):
                yield self.get_single_train_test_split(test_idx)
        else:
            # use the last fold as test set
            yield self.get_single_train_test_split(len(self.folds) - 1)

    def get_single_train_test_split(self, test_idx):
        folds = self.folds
        yearly_data_df, daily_data_df = self.yearly_data_df, self.daily_data_df
        yearly_info_df, daily_info_df = self.yearly_info_df, self.daily_info_df

        test_set = folds[test_idx]
        train_set = []
        for idx, fold in enumerate(folds):
            if idx == test_idx:
                continue
            else:
                train_set.extend(fold)

        yearly_data_train, daily_data_train = yearly_data_df.loc[train_set], daily_data_df.loc[train_set]
        yearly_data_test, daily_data_test = yearly_data_df.loc[test_set], daily_data_df.loc[test_set]

        yearly_info_train, daily_info_train = yearly_info_df.loc[train_set], daily_info_df.loc[train_set]
        yearly_info_test, daily_info_test = yearly_info_df.loc[test_set], daily_info_df.loc[test_set]

        return (
            (yearly_data_train, daily_data_train, yearly_info_train, daily_info_train)
            , (yearly_data_test, daily_data_test, yearly_info_test, daily_info_test)
        )


def sample_and_evaluate(sampler: YearDaySampler, daily_data_train, daily_data_test, yearly_info_test, daily_info_test,
                        nb_samples=100, use_probs=False):
    if use_probs:
        daily_probabilities = sampler.get_sampling_probabilities(yearly_info_test, daily_info_test)
    else:
        daily_probabilities = sampler.generate_samples_and_convert_to_probs(yearly_info_test, daily_info_test,
                                                                            nb_samples)

    energy_scores = calculate_energy_score_for_daily_matrix_daily_consumption_data(daily_probabilities,
                                                                                   daily_data_train, daily_data_test)
    return energy_scores
    #
    # energy_scores = np.zeros(daily_info_test.shape[0])
    # for i in range(daily_info_test.shape[0]):
    #     test_sample_daily_info = daily_info_test
    #     test_sample_info = test_info_df.iloc[i:i + 1]
    #     test_sample_data = test_data_df.iloc[i:i + 1]
    #     daily_probabilities = sampler.old_get_sampling_probabilities_daily(test_sample_info)
    #     energy_score = calculate_energy_score_for_daily_matrix_daily_consumption_data(daily_probabilities,
    #                                                                                   train_data_df,
    #                                                                                   test_sample_data)
    #     energy_scores[i] = energy_score
    # return energy_scores


def sample_and_evaluate_local(sampler: YearDaySampler, daily_data_train, daily_data_test, yearly_info_test,
                              daily_info_test):
    daily_probabilities = sampler.get_sampling_probabilities(yearly_info_test, daily_info_test)
    energy_scores = calculate_energy_score_for_daily_matrix_daily_consumption_data(daily_probabilities,
                                                                                   daily_data_train, daily_data_test)
    return energy_scores
