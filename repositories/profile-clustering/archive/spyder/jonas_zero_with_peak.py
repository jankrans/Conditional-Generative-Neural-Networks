# %% imports
from pathlib import Path

import pandas as pd

from energyclustering import data as data
import archive.visualisation.visualise_ts_clustering as clustering_vis
# %%

measurements_per_day = data.get_timeseries_per_day('Offtake', only_single_meters=True)
master_table = data.get_master_table()
print(measurements_per_day.index.names)

# %%
all_ids = measurements_per_day.reset_index()['iID'].unique()

# choose one of these
# NOTE this one contains some of these peaks
id_to_investigate = all_ids[1]

# id_to_investigate = all_ids[2]

print(f"investigating: {id_to_investigate}")

# %%
# gather the data and put in correct format
df_per_day: pd.DataFrame = measurements_per_day.loc[id_to_investigate]
clustering_vis.show_time_serie(df_per_day, 0, None)

#%%
print("for now we just ignore missing data")
df_per_day.dropna(inplace=True, axis=0)
def check_row(row):
    """
        returns true if there are 'min_nb_of_zeros' consequetive zeros followed by a non zero value
    """
    min_nb_of_zeros = 3
    for i in range(min_nb_of_zeros, len(row)):
        if all(row.iloc[i-min_nb_of_zeros:i]):
            if any(not value for value in row.iloc[i:]):
                return True
    return False

df_zeros = (df_per_day == 0).apply(check_row, axis = 1)
only_zero_peak_days = df_per_day[df_zeros]
figure_dir = Path() / "figures" / "zero_followed_by_peak" / str(id_to_investigate).replace("/", "")
figure_dir.mkdir(parents=True, exist_ok=True)
for index in range(only_zero_peak_days.shape[0]):
    clustering_vis.show_time_serie(only_zero_peak_days, index, figure_dir/f'{only_zero_peak_days.index[index]}.png', 
                                   title=id_to_investigate, ylabel='offtake')