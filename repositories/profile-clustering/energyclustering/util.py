import pandas as pd


def series_to_daily_dataframe(profile, dropna = True):
    daily_df = (
        profile.to_frame('value')
        .assign(
            time=lambda x: add_date(x.index.time),
            date=lambda x: x.index.date.astype('str')
        )
        .pipe(lambda x: pd.pivot_table(x, index='date', columns='time', values='value', dropna= False))
    )
    if dropna:
        daily_df = daily_df.dropna(axis = 0)
    return daily_df


def add_date(series):
    return pd.to_datetime(series, format='%H:%M:%S', exact = False)