import pandas as pd
import holidays
from sklearn.preprocessing import OrdinalEncoder


def preprocess_day_info(day_info):
    ORDINALS = ['season', 'isHoliday', 'isWeekend']
    day_info[ORDINALS] = OrdinalEncoder().fit_transform(day_info[ORDINALS].astype('str'))
    return day_info

def get_day_info_for_dates(dates):
    day_df = pd.DataFrame(index=dates)

    # add standard date information
    day_df = day_df.assign(
        dayOfMonth=lambda x: x.index.day,
        month=lambda x: x.index.month,
        year=lambda x: x.index.year,
        dayOfWeek=lambda x: x.index.weekday,
        isWeekend=lambda x: x.index.day_of_week >= 5,
        dayOfYear=lambda x: x.index.day_of_year,
    )

    # add season info
    day_df['season'] = day_df.index.map(season_from_date)

    # add holiday info
    belgium_holidays = holidays.BE()
    day_df['isHoliday'] = day_df.index.map(lambda date: date in belgium_holidays)
    return day_df


def season_from_date(date):
    month = date.month
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    return 'winter'