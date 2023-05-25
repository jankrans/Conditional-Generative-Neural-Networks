
from datetime import date, datetime, time

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]


def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)

def get_day_type(date):
    if isinstance(date, datetime):
        date = date.date()
    if date.isoweekday() < 6:
        return 'weekday'
    return 'weekend'

def to_datetime(d):
    """
        if d is a time make it into a datetime object on a random date
        if d is a date make it into a datetime object with a random time
        if d is a datetime object do nothing
    """
    if isinstance(d, date):
        t = time.min
        return datetime.combine(d,t)
    if isinstance(d, time):
        t = date(2000, 1, 1)
        return datetime.combine(t,d)
    if isinstance(d, datetime):
        return d
    else:
        raise Exception(f"cannot convert {d} to datetime object")