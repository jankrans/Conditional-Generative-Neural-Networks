from pathlib import Path

import pandas as pd

from energyclustering.data.public import data
from archive.visualisation import visualise_ts_clustering as vis
DATA_DIR = Path().absolute() / 'data'

drg = pd.read_pickle(DATA_DIR / 'READING_2016_preprocessed.pkl')
master_table = data.get_master_table()
ids_with_production = master_table[(master_table['Lokale productie'] == 'Ja') & (master_table['Aantal geïnstalleerde meters'] == 1)].loc[:,"InstallatieID"]
iIDs = ids_with_production
id_to_plot = iIDs.iloc[2]
print(id_to_plot)
# %% Plot example time series:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

iID_inds_to_plot = iIDs[3:4]

#plt.figure()
fig, ax = plt.subplots(figsize = (20,5))
for i, idd in enumerate(iID_inds_to_plot): #each meter ID
    ddd = drg.loc[idd]
    plt.plot(ddd.index, ddd['Offtake'].values, '-', linewidth=0.5)
    plt.xlabel('date&time')
    plt.ylabel('Offtake (kWh)')
    #plt.title('Installation ID: ' + idd)
plt.title(f"Offtake of {iIDs.iloc[3]}")
plt.legend(iID_inds_to_plot, title='Installation ID')
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.show()
#%%
iID_inds_to_plot = iIDs[2:3]

#plt.figure()
fig, ax = plt.subplots(figsize = (20,5))
for i, idd in enumerate(iID_inds_to_plot): #each meter ID
    ddd = drg.loc[idd]
    plt.plot(ddd.index, ddd['Injection'].values, '-', linewidth=0.5)
    plt.xlabel('date&time')
    plt.ylabel('Injection (kWh)')
    #plt.title('Installation ID: ' + idd)
plt.title(f"Injection of {iIDs.iloc[2]}")
plt.legend(iID_inds_to_plot, title='Installation ID')
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.show()

#%%
segment_per_day = data.get_timeseries_per_day("Offtake")
segment_per_day = segment_per_day.loc[id_to_plot]
fig, axes = plt.subplots(figsize = (20,5))
for i in range(0, 3):
    vis.show_time_serie(segment_per_day, i, ax_label = 'Offtake')
    plt.show()

