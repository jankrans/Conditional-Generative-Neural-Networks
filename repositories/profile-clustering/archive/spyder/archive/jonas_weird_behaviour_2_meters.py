#%%
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = Path().absolute()/ 'data'
measurements = pd.read_pickle(DATA_DIR/"READING_2016_preprocessed.pkl")

def get_master_table():
    file_name_master = DATA_DIR / 'master_table_meters_csv.csv'
    data_master = pd.read_csv(file_name_master, sep=',', usecols=range(9), engine='python', encoding = 'utf8')
    return data_master

master_table = get_master_table()
#%%

two_meters = master_table[master_table['Aantal geïnstalleerde meters'] == 2]
two_meters.sort_values('Locatie_ID', inplace=True)
IDs_to_vis = two_meters.iloc[:2,0].to_list()
repr(IDs_to_vis)
#%%
fig, ax = plt.subplots()
for i, idd in enumerate(IDs_to_vis): #each meter ID
    ddd = measurements.loc[idd]
    plt.plot(ddd.index, ddd, '-', linewidth=0.5)
    plt.xlabel('date&time')
    plt.ylabel('usage (kWh)')
    #plt.title('Installation ID: ' + idd)
plt.legend(IDs_to_vis, title='Installation ID')
plt.xticks(rotation=45)
# plt.xlim([datetime.date(2016, 9, 10), datetime.date(2016, 9, 11)])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d%b%y %H:%M"))
plt.show()

#%%
measures_per_day = pd.read_pickle(DATA_DIR/"READING_2016_preprocessed_per_day.pkl")

#%%
zeros = measures_per_day == 0
# for i in range(len(measures_per_day)):
#     if any(zeros.iloc[i,:]):
#         print(measures_per_day.iloc[i,:].values)

print(f"total number of zeros: {zeros.sum().sum()}")

