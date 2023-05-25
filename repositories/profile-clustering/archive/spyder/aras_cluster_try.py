# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:17:04 2020

analysis corrected: now energy production values are subtracted from energy consumption values for each timestamp,
                    there is no need to take the envelope of the signals,
                    units corrected (kW -> kWh)

@author: arasy (Aras Yurtman)
"""
# %% Import data:
from pathlib import Path

import numpy as np
import pandas as pd

# note: this will handle windows paths and linux paths without any problems
from energyclustering.data.public import data

DATA_DIR = Path().absolute() / 'data'

def read_master_table():
    file_name_master = DATA_DIR / 'master-table-meters.csv'
    data_master = pd.read_csv(file_name_master, sep=';', usecols=range(9), engine='python')
    return data_master

def read_reading_table():
    print("reading table...", end = "")
    file_name_reading_zip = DATA_DIR / "READING_2016.CSV"
    data_reading_full = pd.read_csv(file_name_reading_zip, sep=';', parse_dates=[3], dtype={'Meetwaarde':np.float64}, decimal=',')
    print(" DONE ")
    print("preprocessing table...", end="")
    # Organize data:
    iIDs = data_reading_full['InstallatieID'].unique() #unique installationID values

    data_reading_full['Meter read tijdstip-x'] = pd.to_datetime(data_reading_full['Meter read tijdstip'], format="%d%b%y:%H:%M:%S") #convert strings to datetime format

    # Add a signed readings column so that the readings are negative if energy is injected to the system
    data_reading_full['Meetwaarde-signed'] = data_reading_full.apply(lambda o: -o['Meetwaarde'] if o['Meetwaarde']>0 and o['Afname/Injectie']=='Injectie' else o['Meetwaarde'], axis=1)

    # Obtain net readings for each timestamp of each installationID
    drg = data_reading_full.groupby(['InstallatieID', 'Meter read tijdstip-x']).agg(Meetwaarde_net=('Meetwaarde-signed', np.sum))

    # Rename columns as "iID", "datetime", "usage", where the first two are indices (and are sorted)
    drg.index.names = ['iID', 'datetime']
    drg.rename(columns={'Meetwaarde_net': 'usage'}, inplace=True)
    print(" DONE ")
    return iIDs, drg

# iIDs, drg = read_reading_table()
drg = pd.read_pickle(DATA_DIR / 'READING_2016_preprocessed.pkl')
master_table = data.get_master_table()
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]
iIDs = ids_with_production
# %% Plot example time series:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


iID_inds_to_plot = iIDs[2:3]

#plt.figure()
fig, ax = plt.subplots(figsize = (20,5))
for i, idd in enumerate(iID_inds_to_plot): #each meter ID
    ddd = drg.loc[idd]
    plt.plot(ddd.index, ddd['Offtake'].values, '-', linewidth=0.5)
    plt.xlabel('date&time')
    plt.ylabel('Offtake (kWh)')
    #plt.title('Installation ID: ' + idd)
plt.title(f"Offtake of {iIDs.iloc[2]}")
plt.legend(iID_inds_to_plot, title='Installation ID')
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.show()
#%%
iID_inds_to_plot = iIDs[0]

#plt.figure()
fig, ax = plt.subplots()
for i, idd in enumerate(iID_inds_to_plot): #each meter ID
    ddd = drg.loc[idd]
    plt.plot(ddd.index, ddd['Offtake'], '-', linewidth=0.5)
    plt.xlabel('date&time')
    plt.ylabel('offtake (kWh)')
plt.legend(iID_inds_to_plot, title='Installation ID')
plt.xlim([datetime.date(2016, 9, 25), datetime.date(2016, 9, 26)])
plt.ylim([-0.5, 1.5])
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d%b%y %H:%M"))

plt.show()

# %% Plot the DTW warping between two consumption signals in a short time period (one day):
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

datetime_range_to_plot = [datetime.date(2016, 9, 25), datetime.date(2016, 9, 26)]
iID_inds_to_plot = range(74,76)

ixs_to_plot = []
u_to_plot = []
for i, idd in enumerate(iIDs): #each meter ID
    ddd = drg.loc[idd]
    ixs_to_plot.append(np.where((ddd.index >= pd.to_datetime(datetime_range_to_plot[0])) & (ddd.index < pd.to_datetime(datetime_range_to_plot[1]))))
    u_to_plot.append(np.sort(ddd.iloc[ixs_to_plot[i]].values))

s1 = u_to_plot[iID_inds_to_plot[0]][:, 0]
s2 = u_to_plot[iID_inds_to_plot[1]][:, 0]
path = dtw.warping_path(s1, s2, psi=10)
dtwvis.plot_warping(s1, s2, path)

d, paths = dtw.warping_paths(s1, s2, window=12, psi=10) # psi parameter is the P parameter in the following paper (available on Mendeley) (see Table I there):
                                                       # Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. Scientometrics, 26(1), 43–49.
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path)

# %%
from dtaidistance import clustering
series = []
for i, idd in enumerate(iIDs): #each meter ID
    series.append(u_to_plot[i][:, 0])


dtw_distance_matrix = dtw.distance_matrix_fast
# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw_distance_matrix, {})
cluster_idx = model1.old_fit(series)
# Keep track of full tree by using the HierarchicalTree wrapper class
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.old_fit(series)
# You can also pass keyword arguments identical to instantiate a Hierarchical object
model2 = clustering.HierarchicalTree(dists_fun=dtw_distance_matrix, dists_options={})
cluster_idx = model2.old_fit(series)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw_distance_matrix, {})
cluster_idx = model3.old_fit(series)

# model2.plot()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
show_ts_label = lambda idx: "ts-" + str(idx)
model2.plot(axes=ax, show_ts_label=show_ts_label,
           show_tr_label=False, ts_label_margin=-20,
           ts_left_margin=2, ts_sample_length=1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
show_ts_label = lambda idx: "ts-" + str(idx)
model3.plot(axes=ax, show_ts_label=show_ts_label,
           show_tr_label=False, ts_label_margin=-20,
           ts_left_margin=2, ts_sample_length=1)