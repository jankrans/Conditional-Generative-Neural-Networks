import pandas as pd
from pathlib import Path



def read_weather_data(city):
    path = Path(__file__).absolute().parent/'data'/f'weather_data_{city}.pkl'
    return pd.read_pickle(path)