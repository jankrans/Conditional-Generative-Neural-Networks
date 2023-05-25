#%% imports
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import clustering
from dtaidistance import dtw

import matplotlib.dates as mdates
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
id_to_investigate = ids_with_production.iat[0]
print(f"investigating: {id_to_investigate}")
#%%
# check for missing data
df_per_day:pd.DataFrame = measurements_per_day.loc[id_to_investigate]
print(f"Number of (actual) missing data: {df_per_day.isna().sum().sum()}")
print(f"for now lets just ignore days with missing values")
df_per_day.dropna(inplace=True,axis=0)
#%%
# simply plot the days to get an idea of how the data looks
fig, ax = plt.subplots()
# indices_to_plot = [267]
# indices_to_plot = [17,22]
# indices_to_plot = [115, 242, 277, 129,120,224, 147,202, 239,218]
indices_to_plot = [0,12,321,38,349]
for i in indices_to_plot:
    plt.plot([str(i) for i in df_per_day.columns], df_per_day.iloc[i,:].values, label = df_per_day.index[i])
plt.legend()
plt.title("Raw visualisation of all profiles")
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.show()

# there seems to be a clear outlier with huge production!
#%%

series = df_per_day.to_numpy()
print(series.shape)
#%%
# dists = dtw.distance_matrix_fast(series,window = 4)
#%%
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {'window': 4})
# model1.fit(series)
model = clustering.HierarchicalTree(model1)
model.old_fit(series)
#%%
model.plot("test2.png")
#%%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(50, 50))
show_ts_label = lambda idx: "ts-" + str(idx)
show_tr_label = lambda x: str(x)
model.plot(axes=ax, show_ts_label=show_ts_label,
           show_tr_label=False)
fig.show()
#%%
fig.savefig("test.png", dpi = 300, layout ='tight')
#%%


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
model3 = clustering.LinkageTree(dtw_distance_matrix, {}, method='average')
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


