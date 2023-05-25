# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:56:40 2020

@author: arasy
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%
DATA_DIR = Path().absolute()/ 'data'
measurements_per_day = pd.read_pickle(DATA_DIR/"READING_2016_preprocessed_per_day.pkl")

def get_master_table():
    file_name_master = DATA_DIR / 'master_table_meters_csv.csv'
    data_master = pd.read_csv(file_name_master, sep=';', usecols=range(9), engine='python')
    return data_master

master_table = get_master_table()

#%%
# iID's that have local production and that only have a single meter
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]
print(f"Number of normal installations with production: {len(ids_with_production)}")
# choose one of these
id_to_investigate = ids_with_production.iat[1]
print(f"investigating: {id_to_investigate}")
#%%
# check for missing data
df_per_day:pd.DataFrame = measurements_per_day.loc[id_to_investigate]
print(f"Number of (actual) missing data: {df_per_day.isna().sum().sum()}")
print(f"for now lets just ignore days with missing values")
df_per_day.dropna(inplace=True,axis=0)

# %% Poincaré plots:
import pyhrv.nonlinear as nl
d = np.concatenate(df_per_day.values)
nl.poincare(d)

# %% Custom Poincaré plots:
plt.figure()
for i in range(df_per_day.shape[0] - 1):
    plt.plot(df_per_day.iloc[i], df_per_day.iloc[i + 1], '-y', linewidth=0.5)
for i in range(df_per_day.shape[0] - 1):
    plt.plot(df_per_day.iloc[i], df_per_day.iloc[i + 1], 'b.', markersize=1)

