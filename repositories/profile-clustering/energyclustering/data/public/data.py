from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).absolute().parent / 'data'


def get_master_table():
    file_name_master = DATA_DIR / 'raw'/ 'master-table-meters.csv'
    data_master = pd.read_csv(file_name_master, sep=';', usecols=range(9), dtype={'Meetwaarde': np.float64,
                                                                                  'Afname - Injectie': np.float64,
                                                                                  'Jaarlijkse injectie': np.float64,
                                                                                  'Jaarlijkse afname': np.float64,
                                                                                  'Geïnstalleerd vermogen': np.float64},
                              decimal=',', engine='python', encoding='iso8859')
    return data_master

def get_data_reading_full():
    file_name_reading_full = DATA_DIR / "READING_2016_full.pkl"
    data_reading_full = pd.read_pickle(file_name_reading_full , compression='gzip')
    return data_reading_full

def get_data_reading_preprocessed():
    file_name_reading_preprocessed = DATA_DIR / "READING_2016_preprocessed.pkl"
    data_reading_preprocessed = pd.read_pickle(file_name_reading_preprocessed)
    return data_reading_preprocessed

def get_ids(production, only_single_meters):
    master_table = get_master_table()
    production_filter = "Ja" if production else "Nee"
    filter = master_table['Lokale productie'] == production_filter
    if only_single_meters:
        filter = filter & (master_table['Aantal geïnstalleerde meters'] == 1)
    return master_table[filter].loc[:, "InstallatieID"].unique()

def get_timeseries_per_day(value, only_single_meters = True):
    assert value in {"Consumption", "Offtake", "Injection"}
    if only_single_meters:
        path = DATA_DIR / 'per_day_single_meters'
    else:
        path = DATA_DIR / 'per_day'
    return pd.read_pickle(path/ f"READING_2016_{value}_per_day.pkl")

def get_timeseries_per_week(value, only_single_meters = True):
    assert value in {"Consumption", "Offtake", "Injection"}
    if only_single_meters:
        path = DATA_DIR / 'per_week_single_meters'
    else:
        path = DATA_DIR / 'per_week'
    return pd.read_pickle(path/ f"READING_2016_{value}_per_week.pkl")

