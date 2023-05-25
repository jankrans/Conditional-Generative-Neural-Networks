from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pandas as pd

class YearlyInfoPreprocessor:
    def __init__(self, columns_to_use, normalized = True):
        self.scaler = MinMaxScaler() if normalized else None
        self.columns_to_use = columns_to_use

    def fit(self, info_df):
        info = info_df.loc[:, self.columns_to_use]
        if self.scaler is not None:
            self.scaler.fit(info)

    def transform(self, info_df):
        info_df = info_df[self.columns_to_use]
        if self.scaler is not None:
            return pd.DataFrame(self.scaler.transform(info_df), index=info_df.index, columns=info_df.columns)
        return info_df

    def fit_transform(self, info_df):
        self.fit(info_df)
        return self.transform(info_df)

def preprocess_info_baseline(info_df, data_df):

    info_subset = (
        info_df
        .assign(
            # add yearly consumption
            yearly_consumption=data_df.sum(axis = 1)
        )
        # only retain columns that will plausibly be available
        [['connection_power', 'consumer_type', 'PV', 'PV_power', 'yearly_consumption', 'heatpump']]
        .fillna(-1)  # quick fix better preprocessing later
    )

    ORDINALS = ['consumer_type', 'PV', 'PV_power', 'heatpump']
    info_subset[ORDINALS] = OrdinalEncoder().fit_transform(info_subset[ORDINALS].astype('str'))

    return info_subset

def preprocess_info_paper(info_df, data_df):
    info_subset = (
        info_df
        .assign(
            # add yearly consumption
            yearly_consumption = data_df.sum(axis = 1)
        )
        # only retain columns that will be plausibly available
        [['connection_power',  'PV', 'PV_power', 'yearly_consumption']]
        .dropna(subset = ['connection_power'])
        # at this point only the PV_Power column contains NA
        .fillna(-1)
    )
    ORDINALS = ['PV']
    info_subset[ORDINALS] = OrdinalEncoder().fit_transform(info_subset[ORDINALS].astype('str'))

    return info_subset

def combine_household_and_day_info(household_df, day_info):
    day_info = day_info.reset_index()
    query_df = (
        household_df.assign(
            year = lambda x: x.index.map(lambda x: int(x[-5:-1]))
        )
        .reset_index()
        .rename(columns = {'index': 'meterID'})
        .pipe(lambda x: pd.merge(x, day_info.assign(
            year = lambda x: x.date_time.dt.year
        ), on = 'year'))
        .drop(columns = 'year')
        .set_axis(pd.MultiIndex.from_tuples(
            [('household_info', column) for column in ['meterID'] + household_df.columns.tolist() if column != 'year']
            +
            [('day_info', column) for column in day_info.columns if column != 'year']
        ), axis = 1)
        # .set_index([('household_info', 'meterID'), ('day_info', 'date_time')])
        # .rename_axis(['meterID', 'date'], axis = 0)
    )
    query_df.loc[:, ('day_info', 'date_time')] = query_df.loc[:, ('day_info', 'date_time')].apply(lambda x: x.replace(year = 2016))
    query_df = (
        query_df
        .set_index([('household_info', 'meterID'), ('day_info', 'date_time')])
        .rename_axis(['meterID', 'date'], axis = 0)
    )
    # query_df.index = query_df.index.set_levels(query_df.index.levels[1].map(lambda x: x.replace(year = 2016)), level = 1)
    return query_df
