import pandas as pd
import numpy as np
def yearly_profile_df_to_daily_df(df):
    all_dates = pd.date_range('2016-01-01', '2016-12-31')
    all_profiles = df.index
    columns = pd.date_range('2016-01-01 0:00', periods = 96, freq = '15min')
    data = df.to_numpy().reshape((-1, 96))
    return pd.DataFrame(data, index = pd.MultiIndex.from_product([all_profiles, all_dates]), columns = columns)

def filter_nan_days(daily_data_df, daily_info_df, how = 'any'):
    if how == 'any':
        filter = daily_data_df.isna().any(axis = 1)
    elif how == 'all':
        filter = daily_data_df.isna().all(axis = 1)
    else:
        raise Exception()
    return daily_data_df.loc[~filter], daily_info_df.loc[~filter]

def subsample_years(data_df, info_df, nb_profiles, seed = None):
    data_df = data_df.sample(nb_profiles, random_state = seed)
    info_df = info_df.loc[data_df.index]
    return data_df, info_df

def subsample_days(daily_data_df, daily_info_df, week_factor = 1, seed = None):
    generator = np.random.default_rng(seed)

    years = daily_data_df.index.get_level_values(0).unique().map(lambda x: x[-5:-1])
    first_year, last_year = years.min(), years.max()
    all_days = pd.date_range(f"1/1/{first_year}", f"31/12/{last_year}", freq='1D')

    # pick days to sample
    day_df = (
        # start with a dataframe with all the days
        pd.DataFrame(index=all_days)
        # assign some additional information about each day
        .assign(
            year=lambda x: x.index.year,
            is_weekend=lambda x: x.index.day_of_week >= 5,
            iso_week=lambda x: x.index.isocalendar().week,
        )
        # filter weeks using the week factor
        .pipe(lambda x: x[x.iso_week % week_factor == 0])
        .groupby(['year', 'iso_week', 'is_weekend'])
        .apply(lambda x: x.sample(1, random_state = generator))
        .droplevel([0, 1, 2])
        .sort_index()
    )

    # put sampled days in dictionary per year
    sampled_days_per_year = {key:value.apply(lambda x: x.replace(year = 2016)) for key,value in day_df.reset_index().groupby('year')['index']}

    # do the subsampling
    daily_data_df = daily_data_df.groupby(axis = 0, level = 0).apply(_subsample_year, sampled_days_per_year = sampled_days_per_year).droplevel(0)
    daily_info_df = daily_info_df.loc[daily_data_df.index]

    return daily_data_df, daily_info_df


def _subsample_year(df, sampled_days_per_year):
    year = int(df.name[-5:-1])
    days_to_sample = sampled_days_per_year[year]
    return df.loc[(df.name, days_to_sample), :]