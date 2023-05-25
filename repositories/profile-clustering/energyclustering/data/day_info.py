import pandas as pd
import holidays

def construct_day_info_df(first_year = 2011, last_year = 2017):
    # all the dates for which to construct the day_info
    all_days = pd.date_range(f"1/1/{first_year}", f"31/12/{last_year}", freq='1D')

    # dataframe to store the information
    day_df = pd.DataFrame(index=all_days)

    # standard date information
    day_df = day_df.assign(
        day=lambda x: x.index.day,
        month=lambda x: x.index.month,
        year=lambda x: x.index.year,
        day_of_week=lambda x: x.index.weekday,
        is_weekend=lambda x: x.index.day_of_week >= 5,
        iso_day=lambda x: x.index.day_of_year,
    )

    # season information (meteorological seasons)
    day_df['season'] = day_df.index.map(season_from_date)

    # holiday information
    belgium_holidays = holidays.BE()
    day_df['is_holiday'] = all_days.map(lambda date: date in belgium_holidays)

    return day_df 


def season_from_date(date):
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    elif 6<= month <=8:
        return 'summer'
    elif 9<=month <= 11:
        return 'autumn'
    return 'winter'
