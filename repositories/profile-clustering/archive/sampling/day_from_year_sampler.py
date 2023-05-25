
class RandomDayFromYearSampler:
    """
        NOT IN USE
        Samples a random day from a year selected by the yearly sampler

    """
    def __init__(self, yearly_sampler, nb_days=None, seed=1234):
        self.yearly_sampler = yearly_sampler
        self.nb_days = nb_days
        self.random_generator = np.random.default_rng(seed)

        self.daily_data_df = None

    def fit(self, daily_data_df, data_df, daily_info_df):
        household_info = daily_info_df.loc[:, 'household_info'].droplevel('date').pipe(
            lambda x: x[~x.index.duplicated(keep='first')])
        self.yearly_sampler.old_fit(household_info, data_df)

        # save the available dates
        self.daily_data_df = daily_data_df

    def get_sampling_probabilities_daily(self, query_df):
        household_info = query_df.loc[:, 'household_info'].droplevel('date').pipe(
            lambda x: x[~x.index.duplicated(keep='first')])
        sample_probabilities_per_year = self.yearly_sampler.get_sampling_probabilities(household_info)
        sample_probabilities_per_day = []
        for (meterID, date), query_series in query_df.iterrows():
            year_probs = sample_probabilities_per_year.loc[[meterID], :].iloc[0]
            year_probs = year_probs[year_probs > 0]
            daily_probs = (
                # reindex as if you have all the days
                year_probs.reindex(self.daily_data_df.index, level=0)
                .dropna()
                # group by profile
                .groupby(level=0)
            )

            if self.nb_days is None:
                daily_probs = daily_probs.apply(lambda x: x.divide(len(x)))
            else:
                daily_probs = daily_probs.apply(
                    lambda x: x.sample(min(len(x), self.nb_days), random_state=self.random_generator).divide(
                        min(len(x), self.nb_days))).droplevel(0)
            daily_probs = daily_probs.rename((meterID, date))
            sample_probabilities_per_day.append(daily_probs)
        return sample_probabilities_per_day

    def get_sampling_probabilities(self, test_household_info, test_day_info=None):
        sample_probabilities_per_year = self.yearly_sampler.get_sampling_probabilities(test_household_info)
        all_days = pd.date_range(start='2016-01-01', end='2016-12-31')
        daily_index = pd.MultiIndex.from_tuples((profile, year, day) for (profile, year), day in
                                                itertools.product(sample_probabilities_per_year.columns, all_days))
        return sample_probabilities_per_year.divide(len(all_days)).reindex(daily_index, axis=1)


class SpecificDayFromYearSampler:
    """

        NOT IN USE
        A class that is meant to be used on top of a yearly sampler.

        This class first samples a year (using the specified sampler) and then from that year it samples a day

    """

    def __init__(self, yearly_sampler, specific_day):
        self.yearly_sampler = yearly_sampler
        self.specific_day = specific_day

    def fit(self, household_info, day_info, yearly_consumption_data, yearly_clustering, cluster_centroids):
        self.yearly_sampler.old_fit(household_info, yearly_consumption_data, yearly_clustering, cluster_centroids)

    def get_sampling_probabilities(self, test_household_info, test_day_info):
        sample_probabilities_per_year = self.yearly_sampler.get_sampling_probabilities(test_household_info)

        daily_index = pd.MultiIndex.from_tuples(
            (profile, year, self.specific_day) for profile, year in sample_probabilities_per_year.columns)
        # from sample probabilities per year to sample probabilities per day
        sample_probabilities_per_day = (
            sample_probabilities_per_year
            .reindex(daily_index, axis=1)
        )
        return sample_probabilities_per_day



        #
        # samples_to_take = self.yearly_consumption_data.shape[0] if self.n_samples is None else min(
        #     self.yearly_consumption_data.shape[0], self.n_samples)
        # for test_day in test_info.index:
        #
        #     self.ev_filter.fit(self.daily_info_df.loc[sample], None)
        #     sampling_probabilities = self.ev_filter.get_sampling_probabilities_daily(test_info.loc[[test_day], :])
        #     result.
        #
        #
        #     probabilities = np.full((samples_to_take,), 1 / samples_to_take)
        #     probability_series = pd.Series(probabilities, index=sample.index, name=test_day)
        #     result.append(probability_series)
        # return result


class IndividualDailySamplerFromClusterSampler:
    """
        NOT IN USE
    """
    def __init__(self, yearly_sampler, daily_sampler):
        self.yearly_sampler = yearly_sampler
        self.daily_sampler_prototype = daily_sampler
        self.daily_samplers_per_cluster = None

    def fit(self, daily_consumption_data, yearly_consumption_data, daily_info_df):
        # get household info
        household_info = daily_info_df.loc[:, 'household_info'].droplevel('date').pipe(
            lambda x: x[~x.index.duplicated(keep='first')])

        # fit the yearly sampler
        self.yearly_sampler.old_fit(household_info, yearly_consumption_data)
        clustering = self.yearly_sampler.clustering

        # for each profile in each cluster fit a daily sampler
        self.daily_samplers_per_cluster = defaultdict(list)
        for cluster_idx, cluster_df in clustering.groupby(clustering):
            profiles_in_cluster = cluster_df.index
            profile_ids_per_id = clustering.index.to_frame()[0].apply(lambda x: x[2:-8]).to_frame(
                'ID').reset_index().set_index('ID')
            for meterID, profile_IDs in profile_ids_per_id.groupby('index'):
                profiles = profile_IDs.iloc[:, 0]
                all_days = daily_consumption_data.loc[profiles]
                day_info = daily_info_df.loc[profiles, 'day_info']
                profile_sampler = copy.deepcopy(self.daily_sampler_prototype)
                profile_sampler.old_fit(day_info, all_days)
                self.daily_samplers_per_cluster[cluster_idx].append(profile_sampler)

    def get_sampling_probabilities_daily(self, query_df):
        # drop_duplicates because for the same household there might be multiple day_info's
        unique_household_info = query_df.loc[:, 'household_info'].drop_duplicates().droplevel(1)
        cluster_probabilities_per_household = (
            pd.DataFrame(self.yearly_sampler.get_cluster_probabilities(unique_household_info),
                         index=unique_household_info.index)
        )

        sample_probabilities_per_day_collection = []
        for (test_meterID, test_date), query_series in query_df.iterrows():
            household_info = query_series.loc['household_info']
            day_info = query_series.loc['day_info']

            cluster_probs = cluster_probabilities_per_household.loc[[test_meterID], :].iloc[0]
            non_zero_clusters, = np.where(cluster_probs > 0)

            daily_sampling_probs_for_profile = []
            for cluster in non_zero_clusters:
                # get sampling probabilities for all the daily samplers
                daily_samplers = self.daily_samplers_per_cluster[cluster]
                all_sampling_probs = [sampler.old_get_sampling_probabilities_daily(day_info.to_frame().T)[0] for sampler in
                                      daily_samplers]

                # transform to sampling probs per cluster
                daily_sampling_probs_in_cluster = pd.concat(all_sampling_probs, axis=0) * cluster_probs[cluster] / len(
                    daily_samplers)

                # append to sampling probs for profile
                daily_sampling_probs_for_profile.append(daily_sampling_probs_in_cluster)

            daily_sampling_probs_for_profile = pd.concat(daily_sampling_probs_for_profile)
            sample_probabilities_per_day_collection.append(daily_sampling_probs_for_profile)

        return sample_probabilities_per_day_collection



class SimilarDayFromYearSampler:
    """
        NOT IN USE
    """
    def __init__(self, yearly_sampler, n_similar_days_to_consider, weather_info):
        self.yearly_sampler = yearly_sampler
        self.n_similar_days_to_consider = n_similar_days_to_consider

        self.weather_info = weather_info
        self.yearly_consumption_data = None
        self.daily_consumption_data = None

    def fit(self, daily_consumption_data, yearly_consumption_data, daily_info_df):
        household_info = daily_info_df.loc[:, 'household_info'].droplevel('date').pipe(
            lambda x: x[~x.index.duplicated(keep='first')])
        # weather_info = daily_info_df.loc[:, 'day_info'].droplevel('meterID').pipe(lambda x: x[~x.index.duplicated(keep = 'first')])

        self.daily_consumption_data = daily_consumption_data

        # fit the yearly sampler
        self.yearly_sampler.old_fit(household_info, yearly_consumption_data)

        # save weather info
        # self.weather_info = weather_info
        self.yearly_consumption_data = yearly_consumption_data

    def get_sampling_probabilities_daily(self, query_df):
        return self.get_sampling_probabilities(query_df)

    def get_sampling_probabilities(self, query_df):
        """
            test_household_day_info is a dataframe where each row is a query, columns are (household_info, ... ) and (day_info, ...)
        """
        # drop_duplicates because for the same household there might be multiple day_info's
        unique_household_info = query_df.loc[:, 'household_info'].drop_duplicates().droplevel(1)
        sample_probabilities_per_household = (
            self.yearly_sampler.get_sampling_probabilities(unique_household_info)
        )

        # drop the training households that have a sampling probability of zero everywhere
        non_zero_columns = (sample_probabilities_per_household > 0).any(axis=0)
        sample_probabilities_per_household = sample_probabilities_per_household.loc[:, non_zero_columns]

        idx = pd.IndexSlice
        sample_probabilities_per_day_collection = []
        for (testMeterID, test_date), query_series in query_df.iterrows():
            household_info = query_series.loc['household_info']
            # only use the columns that are available in the training_weather_data
            day_info = query_series.loc['day_info', self.weather_info.columns]

            # get the sampling probabilities for this household
            yearly_sample_probabilities = sample_probabilities_per_household.loc[[testMeterID], :].iloc[0]
            yearly_sample_probabilities = yearly_sample_probabilities[yearly_sample_probabilities > 0]

            all_tuples = []
            for meterID in yearly_sample_probabilities.index:
                # determine the available days
                year = int(meterID[-5:-1])
                available_days = self.daily_consumption_data.loc[meterID].index.map(lambda x: x.replace(year=year))

                # determine the closest days
                if len(available_days) <= self.n_similar_days_to_consider:
                    closest_weather_days = available_days
                else:
                    train_weather_info = self.weather_info.loc[available_days]
                    # print(day_info.index)
                    # print(train_weather_info.columns)
                    closest_weather_days = self.get_closest_weather_days(day_info, train_weather_info)

                all_tuples.extend((meterID, day.replace(year=2016)) for day in closest_weather_days)

            desired_columns = pd.MultiIndex.from_tuples(all_tuples)
            daily_sample_probabilities = yearly_sample_probabilities.reindex(desired_columns, axis=0, level=0).groupby(
                axis=0, level=0).apply(lambda x: x.divide(x.shape[0])).rename((testMeterID, test_date))

            sample_probabilities_per_day_collection.append(daily_sample_probabilities)

        # concatenating the result takes way to much time!
        # return pd.concat(sample_probabilities_per_day_collection, axis = 0)
        # so just return a list of the dataframes!

        return sample_probabilities_per_day_collection

    def get_sampling_probabilities_same_days_every_year(self, query_df):
        """
            test_household_day_info is a dataframe where each row is a query, columns are (household_info, ... ) and (day_info, ...)
        """
        # drop_duplicates because for the same household there might be multiple day_info's
        unique_household_info = query_df.loc[:, 'household_info'].drop_duplicates().droplevel(1)
        sample_probabilities_per_household = (
            self.yearly_sampler.get_sampling_probabilities(unique_household_info)
        )

        # drop the training households that have a sampling probability of zero everywhere
        non_zero_columns = (sample_probabilities_per_household > 0).any(axis=0)
        sample_probabilities_per_household = sample_probabilities_per_household.loc[:, non_zero_columns]

        idx = pd.IndexSlice
        sample_probabilities_per_day_collection = []
        for (testMeterID, test_date), query_series in query_df.iterrows():
            household_info = query_series.loc['household_info']
            day_info = query_series.loc['day_info']

            # get the sampling probabilities for this household
            yearly_sample_probabilities = sample_probabilities_per_household.loc[[testMeterID], :].iloc[0]
            yearly_sample_probabilities = yearly_sample_probabilities[yearly_sample_probabilities > 0]

            # get the years of the possible matches
            years = yearly_sample_probabilities.index.map(lambda x: int(x[-5:-1])).unique()

            # for each year calculate the closest days to the test day
            closest_weather_days_per_year = self._get_closest_weather_days_per_year(years, day_info)

            all_tuples = []
            for meterID in yearly_sample_probabilities.index:
                year = int(meterID[-5:-1])
                if year in closest_weather_days_per_year:
                    closest_weather_days = closest_weather_days_per_year[year]
                else:
                    closest_year = min(closest_weather_days_per_year.index, key=lambda x: abs(x - year))
                    closest_weather_days = closest_weather_days_per_year[closest_year]
                all_tuples.extend((meterID, day.replace(year=2016)) for day in closest_weather_days)

            desired_columns = pd.MultiIndex.from_tuples(all_tuples).intersection(self.daily_consumption_data.index,
                                                                                 sort=None)
            missing_data = desired_columns.difference(self.daily_consumption_data.index)
            assert len(missing_data) == 0

            daily_sample_probabilities = yearly_sample_probabilities.reindex(desired_columns, axis=0, level=0).groupby(
                axis=0, level=0).apply(lambda x: x.divide(x.shape[0])).rename((testMeterID, test_date))

            sample_probabilities_per_day_collection.append(daily_sample_probabilities)

        # concatenating the result takes way to much time!
        # return pd.concat(sample_probabilities_per_day_collection, axis = 0)
        # so just return a list of the dataframes!

        return sample_probabilities_per_day_collection

    def get_sampling_probabilities_single_weather(self, test_household_info, test_day_info):
        # first sample a year
        # df with index test_households and columns training_households
        sample_probabilities_per_year = (
            self.yearly_sampler.get_sampling_probabilities(test_household_info)

        )

        # drop the training households that have a sampling probability of zero everywhere
        non_zero_columns = (sample_probabilities_per_year > 0).any(axis=0)
        sample_probabilities_per_year = sample_probabilities_per_year.loc[:, non_zero_columns]

        # get the years that we sample from
        sampled_years = sample_probabilities_per_year.index.get_level_values('year').unique()

        # for each sampled year calculate the x closest days to the test_weather_info
        closest_weather_days_per_year = self._get_closest_weather_days_per_year(sampled_years, test_day_info)

        # desired index
        daily_index = pd.MultiIndex.from_tuples(
            (profile, year, date) for (profile, year) in sample_probabilities_per_year for date in
            closest_weather_days_per_year[year])
        sample_probabilities_per_day = (
            sample_probabilities_per_year
            .divide(self.n_similar_days_to_consider)
            .reindex(daily_index, axis=1)
        )

        return sample_probabilities_per_day

    def get_closest_weather_days(self, test_weather_info, train_weather_info):
        return _select_most_similar_weather_days_for_year(train_weather_info, test_weather_info,
                                                          self.n_similar_days_to_consider)

    def _get_closest_weather_days(self, years, test_weather_info: pd.Series):
        train_weather_info = self.weather_info

        # only consider data where we need to sample from
        train_weather_info = train_weather_info[train_weather_info.index.get_level_values('year').isin(years)]

        # find the closest days
        return _select_most_similar_weather_days_for_year(train_weather_info, test_weather_info,
                                                          self.n_similar_days_to_consider)

    def _get_closest_weather_days_per_year(self, years, test_weather_info: pd.Series):
        """
            test_weather_info is a series with values for several weather attributes
        """
        train_weather_info = self.weather_info

        # only consider data where we need to sample from
        train_weather_info = train_weather_info[train_weather_info.index.year.isin(years)]

        # train weather_info is a dataframe, index is every day for every training year, columns are weather attributes
        closest_training_days_per_year = train_weather_info.groupby(train_weather_info.index.year).apply(
            _select_most_similar_weather_days_for_year, test_weather_info=test_weather_info,
            n_closest=self.n_similar_days_to_consider)
        return closest_training_days_per_year


def _select_most_similar_weather_days_for_year(year_df, test_weather_info, n_closest):
    distances = euclidean_distances(year_df.to_numpy(), np.array([test_weather_info.to_numpy()])).ravel()
    n_closest_idxs = np.argpartition(distances, n_closest)[:n_closest]
    return year_df.index[n_closest_idxs]
