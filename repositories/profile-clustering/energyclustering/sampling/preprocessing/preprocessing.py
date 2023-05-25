"""
    Some additional preprocessing for the info_df in this notebook
"""
import pandas as pd

from energyclustering.data.fluvius import read_data_pickle
from energyclustering.data.weather.data import read_weather_data
from energyclustering.sampling.preprocessing.data import yearly_profile_df_to_daily_df, filter_nan_days, subsample_days, \
    subsample_years
from energyclustering.sampling.preprocessing.day_info import get_day_info_for_dates, preprocess_day_info
from energyclustering.sampling.preprocessing.info import preprocess_info_baseline, preprocess_info_paper, \
    combine_household_and_day_info
from energyclustering.sampling.preprocessing.weather import preprocess_weather_baseline, preprocess_weather_paper

INFO_DF_PREPROCESSING = dict(
    baseline=preprocess_info_baseline,
    paper=preprocess_info_paper,
)
WEATHER_DF_PREPROCESSING = dict(
    baseline=preprocess_weather_baseline,
    paper=preprocess_weather_paper,

)

DAY_DF_PREPROCESSING = dict(
    baseline=preprocess_day_info
)


class DataPreprocessor:

    def __init__(self, data_df=None, info_df=None, weather_df=None):
        if data_df is None:
            info_df, data_df = read_data_pickle()

        if weather_df is None:
            weather_df = read_weather_data('brussels')

        self.raw_info_df = info_df
        self.raw_data_df = data_df
        self.raw_weather_df = weather_df

        # yearly preprocessing
        self.subsample_years_nb = None

        # info_df preprocessing
        self.info_preprocessing_strategy = 'baseline'

        # weather_df preprocessing
        self.weather_preprocessing_strategy = 'baseline'

        # day_df preprocessing
        self.day_preprocessing_strategy = 'baseline'

        # daily preprocessing
        self.drop_nan_days = True
        self.subsample_days_weekfactor = None

    def subsample_years(self, nb_of_profiles):
        self.subsample_years_nb = nb_of_profiles
        return self

    def subsample_days(self, week_reduction_factor):
        self.subsample_days_weekfactor = week_reduction_factor
        return self

    def drop_days_with_nan(self, drop_nan):
        self.drop_nan_days = drop_nan
        return self

    def preprocess_day_df(self, strategy):
        self.day_preprocessing_strategy = strategy
        return self

    def preprocess_info_df(self, strategy):
        self.info_preprocessing_strategy = strategy
        return self

    def preprocess_weather_df(self, strategy):
        self.weather_preprocessing_strategy = strategy
        return self

    def get_data(self):
        weather_df = self.raw_weather_df

        yearly_info_df = self.raw_info_df.set_axis(self.raw_info_df.index.to_flat_index().map(str), axis=0)
        yearly_data_df = self.raw_data_df.set_axis(self.raw_data_df.index.to_flat_index().map(str), axis=0)

        # preprocess info df
        yearly_info_df = INFO_DF_PREPROCESSING[self.info_preprocessing_strategy](yearly_info_df, yearly_data_df)

        # make sure that if some consumers were removed they are removed from the data_df as well
        yearly_data_df = yearly_data_df.loc[yearly_info_df.index]

        # subsample if necessary
        if self.subsample_years_nb is not None:
            yearly_data_df, yearly_info_df = subsample_years(yearly_data_df, yearly_info_df, self.subsample_years_nb, seed=123)

        # preprocess weather df and day_df
        weather_df = WEATHER_DF_PREPROCESSING[self.weather_preprocessing_strategy](weather_df)

        # get day info + preprocess
        calendar_info = get_day_info_for_dates(weather_df.index)
        calendar_info = DAY_DF_PREPROCESSING[self.day_preprocessing_strategy](calendar_info)

        # add weather and day info
        full_day_info = pd.concat([weather_df, calendar_info], axis=1)

        # combine day and household info full_day_info
        daily_info_df = combine_household_and_day_info(yearly_info_df, full_day_info).loc[:, 'day_info']

        # yearly data to daily data
        daily_data_df = yearly_profile_df_to_daily_df(yearly_data_df)

        # filter NaN days if necessary
        if self.drop_nan_days:
            daily_data_df, daily_info_df = filter_nan_days(daily_data_df, daily_info_df)

        if self.subsample_days_weekfactor is not None:
            daily_data_df, daily_info_df = subsample_days(daily_data_df, daily_info_df,
                                                          week_factor=self.subsample_days_weekfactor, seed=123)

        return daily_data_df.sort_index(), yearly_data_df.sort_index(), daily_info_df.sort_index(), yearly_info_df.sort_index()

    def old_get_data(self):
        weather_df = self.raw_weather_df

        info_df = self.raw_info_df.set_axis(self.raw_info_df.index.to_flat_index().map(str), axis=0)
        data_df = self.raw_data_df.set_axis(self.raw_data_df.index.to_flat_index().map(str), axis=0)

        # preprocess info df
        info_df = INFO_DF_PREPROCESSING[self.info_preprocessing_strategy](info_df, data_df)

        # make sure that if some consumers were removed they are removed from the data_df as well
        data_df = data_df.loc[info_df.index]

        # subsample if necessary
        if self.subsample_years_nb is not None:
            data_df, info_df = subsample_years(data_df, info_df, self.subsample_years_nb, seed=123)

        # preprocess weather df and day_df
        weather_df = WEATHER_DF_PREPROCESSING[self.weather_preprocessing_strategy](weather_df)

        # get day info + preprocess
        day_info = get_day_info_for_dates(weather_df.index)
        day_info = DAY_DF_PREPROCESSING[self.day_preprocessing_strategy](day_info)

        # add weather and day info
        full_day_info = pd.concat([weather_df, day_info], axis=1)

        # combine day and household info daily_info_df
        daily_info_df = combine_household_and_day_info(info_df, full_day_info)

        # yearly data to daily data
        daily_data_df = yearly_profile_df_to_daily_df(data_df)

        # filter NaN days if necessary
        if self.drop_nan_days:
            daily_data_df, daily_info_df = filter_nan_days(daily_data_df, daily_info_df)

        if self.subsample_days_weekfactor is not None:
            daily_data_df, daily_info_df = subsample_days(daily_data_df, daily_info_df,
                                                          week_factor=self.subsample_days_weekfactor, seed=123)

        return daily_data_df, data_df, daily_info_df, weather_df


def to_daily_metadata_df(household_df, day_info):
    day_info = day_info.reset_index()
    query_df = (
        household_df.assign(
            year=lambda x: x.index.map(lambda x: int(x[-5:-1]))
        )
        .reset_index()
        .rename(columns={'index': 'meterID'})
        .pipe(lambda x: pd.merge(x, day_info.assign(
            year=lambda x: x.date_time.dt.year
        ), on='year'))
        .drop(columns='year')
        .set_axis(pd.MultiIndex.from_tuples(
            [('household_info', column) for column in ['meterID'] + household_df.columns.tolist() if column != 'year']
            +
            [('day_info', column) for column in day_info.columns if column != 'year']
        ), axis=1)
        # .set_index([('household_info', 'meterID'), ('day_info', 'date_time')])
        # .rename_axis(['meterID', 'date'], axis = 0)
    )
    query_df.loc[:, ('day_info', 'date_time')] = query_df.loc[:, ('day_info', 'date_time')].apply(
        lambda x: x.replace(year=2016))
    query_df = (
        query_df
        .set_index([('household_info', 'meterID'), ('day_info', 'date_time')])
        .rename_axis(['meterID', 'date'], axis=0)
    )
    # query_df.index = query_df.index.set_levels(query_df.index.levels[1].map(lambda x: x.replace(year = 2016)), level = 1)
    return query_df
