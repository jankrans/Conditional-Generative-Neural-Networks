from energyclustering import data as data
import pandas as pd
import numpy as np
import archive.visualisation.visualise_ts_clustering as vis_clust

# get data for houses with no production
master_table = data.get_master_table()
ids_without_production = master_table[(master_table['Lokale productie'] == 'Nee')].loc[:,"InstallatieID"].unique()
df:pd.DataFrame = data.get_timeseries_per_day("Consumption")
df = df[df.index.isin(ids_without_production, level = 0)]

def investigate_household_month_per_day(household_id, months):
      household_df = df.loc[household_id]
      household_df = household_df[[date.month in months for date in household_df.index]]
      print(household_df.head())
      cluster_labeling = [date.weekday() for date in household_df.index]
      cluster_dict = vis_clust.cluster_labeling_to_dict(cluster_labeling)
      vis_clust.visualise_ts_clusters_per_day(cluster_dict, household_df)

def investigate_household_month_per_weekday_vs_weekend(household_id, months):
      household_df = df.loc[household_id]
      household_df = household_df[[date.month in months for date in household_df.index]]
      print(household_df.head())
      cluster_labeling = [1 if date.weekday() < 5 else 2 for date in household_df.index]
      cluster_dict = vis_clust.cluster_labeling_to_dict(cluster_labeling)
      vis_clust.visualise_ts_clusters_per_day(cluster_dict, household_df)

def investigate_household_month_per_weekday_vs_weekend_clustering(household_id, months):
      household_df = df.loc[household_id]
      household_df = household_df[[date.month in months for date in household_df.index]]
      print(household_df.head())
      cluster_labeling = np.array([1 if date.weekday() < 5 else 2 for date in household_df.index])
      weekday_household_df = household_df[cluster_labeling==1]
      weekend_household_df = household_df[cluster_labeling==2]
      cluster_dict = vis_clust.cluster_labeling_to_dict(cluster_labeling)
      vis_clust.visualise_ts_clusters_per_day(cluster_dict, household_df)

investigate_household_month_per_weekday_vs_weekend(ids_without_production[61], range(4,6))




